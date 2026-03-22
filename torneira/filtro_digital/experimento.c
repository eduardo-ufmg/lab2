#define _GNU_SOURCE

#include <NIDAQmx.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>

#define SAMPLING_PERIOD_SEC 0.1
#define SAMPLES_PER_SECOND (1.0 / SAMPLING_PERIOD_SEC)
#define N_SAMPLES 5400
#define RT_PRIORITY 90

/* Filter Parameters */
#define PLANT_DELAY_L 1.043
#define DELAY_FRACTION 0.15
#define MEDIAN_N 5

#define DAQmxErrChk(functionCall)                                                                  \
    do {                                                                                           \
        int32 error = (functionCall);                                                              \
        if (DAQmxFailed(error)) {                                                                  \
            char errBuff[2048] = {'\0'};                                                           \
            DAQmxGetExtendedErrorInfo(errBuff, 2048);                                              \
            fprintf(stderr, "DAQmx Error [%d]: %s\n", error, errBuff);                             \
            goto Error;                                                                            \
        }                                                                                          \
    } while (0)

typedef struct
{
    int k;
    float64 u;
    float64 y;
    float64 y_filtered;
} sample_t;

typedef struct
{
    float64 alpha;
    float64 feedforward_coeff;
    float64 y_prev;
    float64 x_buffer[MEDIAN_N];
    int buffer_idx;
    int is_initialized;
} hybrid_filter_t;

TaskHandle AItaskHandle = 0;
TaskHandle AOtaskHandle = 0;
sample_t record_buffer[N_SAMPLES];

static void filter_init(hybrid_filter_t * f, float64 L, float64 Ts, float64 delay_frac,
                        float64 initial_state)
{
    float64 tau_f = L * delay_frac;
    f->alpha = exp(-Ts / tau_f);
    f->feedforward_coeff = 1.0 - f->alpha;
    f->y_prev = initial_state;
    f->buffer_idx = 0;
    f->is_initialized = 1;

    for (int i = 0; i < MEDIAN_N; ++i) {
        f->x_buffer[i] = initial_state;
    }
}

static float64 get_median(const float64 * buffer)
{
    float64 temp[MEDIAN_N];
    memcpy(temp, buffer, sizeof(float64) * MEDIAN_N);

    /* Deterministic Insertion Sort */
    for (int i = 1; i < MEDIAN_N; i++) {
        float64 key = temp[i];
        int j = i - 1;
        while (j >= 0 && temp[j] > key) {
            temp[j + 1] = temp[j];
            j = j - 1;
        }
        temp[j + 1] = key;
    }
    return temp[MEDIAN_N / 2];
}

static float64 filter_step(hybrid_filter_t * f, float64 x)
{
    f->x_buffer[f->buffer_idx] = x;
    f->buffer_idx = (f->buffer_idx + 1) % MEDIAN_N;

    float64 x_med = get_median(f->x_buffer);
    float64 y = f->alpha * f->y_prev + f->feedforward_coeff * x_med;
    f->y_prev = y;

    return y;
}

static int init_board(void)
{
    DAQmxErrChk(DAQmxCreateTask("AI_Task", &AItaskHandle));
    DAQmxErrChk(DAQmxCreateTask("AO_Task", &AOtaskHandle));

    DAQmxErrChk(DAQmxCreateAIVoltageChan(AItaskHandle, "Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0,
                                         10.0, DAQmx_Val_Volts, NULL));
    DAQmxErrChk(
        DAQmxCreateAOVoltageChan(AOtaskHandle, "Dev1/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, NULL));

    DAQmxErrChk(DAQmxCfgSampClkTiming(AItaskHandle, "", SAMPLES_PER_SECOND, DAQmx_Val_Rising,
                                      DAQmx_Val_HWTimedSinglePoint, 1));
    DAQmxErrChk(DAQmxCfgSampClkTiming(AOtaskHandle, "", SAMPLES_PER_SECOND, DAQmx_Val_Rising,
                                      DAQmx_Val_HWTimedSinglePoint, 1));

    DAQmxErrChk(
        DAQmxSetRealTimeWaitForNextSampClkWaitMode(AItaskHandle, DAQmx_Val_WaitForInterrupt));
    DAQmxErrChk(DAQmxSetRealTimeConvLateErrorsToWarnings(AItaskHandle, 1));

    DAQmxErrChk(DAQmxStartTask(AOtaskHandle));
    DAQmxErrChk(DAQmxStartTask(AItaskHandle));

    return 0;

Error:
    if (AItaskHandle != 0) {
        DAQmxStopTask(AItaskHandle);
        DAQmxClearTask(AItaskHandle);
        AItaskHandle = 0;
    }
    if (AOtaskHandle != 0) {
        DAQmxStopTask(AOtaskHandle);
        DAQmxClearTask(AOtaskHandle);
        AOtaskHandle = 0;
    }
    return -1;
}

void * control_loop_task(void * arg)
{
    (void)arg;
    float64 data_read = 0.0;
    float64 data_write = 8.0;
    float64 data_filtered = 0.0;

    hybrid_filter_t filter = {0};

    for (int sample_ctr = 0; sample_ctr < N_SAMPLES; ++sample_ctr) {

        if (sample_ctr > 3600) {
            data_write = 8.0;
        } else if (sample_ctr > 1800) {
            data_write = 2.0;
        }

        int32 err = DAQmxWriteAnalogScalarF64(AOtaskHandle, 1, 10.0, data_write, NULL);
        if (DAQmxFailed(err)) {
            fprintf(stderr, "RT Loop AO Write Failed: %d\n", err);
            break;
        }

        err = DAQmxWaitForNextSampleClock(AItaskHandle, 10.0, NULL);
        if (DAQmxFailed(err)) {
            fprintf(stderr, "RT Loop Wait For Clock Failed: %d\n", err);
            break;
        }

        err = DAQmxReadAnalogScalarF64(AItaskHandle, 10.0, &data_read, NULL);
        if (DAQmxFailed(err)) {
            fprintf(stderr, "RT Loop AI Read Failed: %d\n", err);
            break;
        }

        if (!filter.is_initialized) {
            filter_init(&filter, PLANT_DELAY_L, SAMPLING_PERIOD_SEC, DELAY_FRACTION, data_read);
        }

        data_filtered = filter_step(&filter, data_read);

        record_buffer[sample_ctr].k = sample_ctr;
        record_buffer[sample_ctr].u = data_write;
        record_buffer[sample_ctr].y = data_read;
        record_buffer[sample_ctr].y_filtered = data_filtered;
    }

    return NULL;
}

int main(void)
{
    struct sched_param param;
    pthread_attr_t attr;
    pthread_t thread;

    if (init_board() < 0) {
        fprintf(stderr, "Hardware initialization failed. Aborting.\n");
        return EXIT_FAILURE;
    }

    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("mlockall failed");
        goto cleanup;
    }

    if (pthread_attr_init(&attr)) {
        perror("pthread_attr_init failed");
        goto cleanup;
    }

    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);

    param.sched_priority = RT_PRIORITY;
    pthread_attr_setschedparam(&attr, &param);

    if (pthread_create(&thread, &attr, control_loop_task, NULL)) {
        perror("pthread_create failed");
        goto cleanup;
    }

    pthread_join(thread, NULL);

    FILE * exp_data_f = fopen("experimento.txt", "w");
    if (exp_data_f) {
        fprintf(exp_data_f, "k, u, y, y_filtered\n");
        for (int i = 0; i < N_SAMPLES; ++i) {
            if (i > 0 && record_buffer[i].k == 0 && record_buffer[i].u == 0.0)
                break;
            fprintf(exp_data_f, "%d, %lf, %lf, %lf\n", record_buffer[i].k, record_buffer[i].u,
                    record_buffer[i].y, record_buffer[i].y_filtered);
        }
        fclose(exp_data_f);
    } else {
        perror("Failed to open file for writing");
    }

cleanup:
    if (AItaskHandle) {
        DAQmxStopTask(AItaskHandle);
        DAQmxClearTask(AItaskHandle);
    }
    if (AOtaskHandle) {
        DAQmxStopTask(AOtaskHandle);
        DAQmxClearTask(AOtaskHandle);
    }

    return EXIT_SUCCESS;
}
