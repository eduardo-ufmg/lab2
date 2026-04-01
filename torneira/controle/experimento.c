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
#define MEDIAN_SAMPLES 5
#define ALPHA 0.88

/* Thread-safe error handling macro */
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
    float64 y_raw;
    float64 y_hat;
} sample_t;

typedef struct
{
    float64 y_hist[MEDIAN_SAMPLES];
    float64 y_hat_prev;
    int is_initialized;
} filter_state_t;

static inline void filter_init(filter_state_t * f)
{
    memset(f->y_hist, 0, sizeof(f->y_hist));
    f->y_hat_prev = 0.0;
}

static inline float64 filter_update(filter_state_t * f, float64 y_n)
{
    float64 temp_hist[MEDIAN_SAMPLES];

    /* Shift history register */
    for (int i = MEDIAN_SAMPLES - 1; i > 0; --i) {
        f->y_hist[i] = f->y_hist[i - 1];
    }
    f->y_hist[0] = y_n;

    /* Copy and sort for median extraction (Inline Insertion Sort) */
    memcpy(temp_hist, f->y_hist, sizeof(temp_hist));
    for (int i = 1; i < MEDIAN_SAMPLES; i++) {
        float64 key = temp_hist[i];
        int j = i - 1;
        while (j >= 0 && temp_hist[j] > key) {
            temp_hist[j + 1] = temp_hist[j];
            j = j - 1;
        }
        temp_hist[j + 1] = key;
    }

    float64 x_n = temp_hist[MEDIAN_SAMPLES / 2];

    /* Exponential Moving Average (EMA) */
    float64 y_hat = ALPHA * f->y_hat_prev + (1.0 - ALPHA) * x_n;

    f->y_hat_prev = y_hat;
    return y_hat;
}

/* Global state: locked in RAM via mlockall */
TaskHandle AItaskHandle = 0;
TaskHandle AOtaskHandle = 0;
sample_t record_buffer[N_SAMPLES];

static int init_board(void)
{
    DAQmxErrChk(DAQmxCreateTask("AI_Task", &AItaskHandle));
    DAQmxErrChk(DAQmxCreateTask("AO_Task", &AOtaskHandle));

    DAQmxErrChk(DAQmxCreateAIVoltageChan(AItaskHandle, "Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0,
                                         10.0, DAQmx_Val_Volts, NULL));
    DAQmxErrChk(
        DAQmxCreateAOVoltageChan(AOtaskHandle, "Dev1/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, NULL));

    /* Configure for Hardware Timed Single Point (HWTSP) */
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

    filter_state_t state_estimator;
    filter_init(&state_estimator);

    for (int sample_ctr = 0; sample_ctr < N_SAMPLES; ++sample_ctr) {

        /* Control law / Trajectory generation */
        if (sample_ctr > 3600) {
            data_write = 8.0;
        } else if (sample_ctr > 1800) {
            data_write = 2.0;
        }

        /* 1. Write DAC (buffered for next clock edge) */
        int32 err = DAQmxWriteAnalogScalarF64(AOtaskHandle, 1, 10.0, data_write, NULL);
        if (DAQmxFailed(err)) {
            fprintf(stderr, "RT Loop AO Write Failed: %d\n", err);
            break;
        }

        /* 2. Block until hardware sample clock edge */
        err = DAQmxWaitForNextSampleClock(AItaskHandle, 10.0, NULL);
        if (DAQmxFailed(err)) {
            fprintf(stderr, "RT Loop Wait For Clock Failed: %d\n", err);
            break;
        }

        /* 3. Read ADC (latched at clock edge) */
        err = DAQmxReadAnalogScalarF64(AItaskHandle, 10.0, &data_read, NULL);
        if (DAQmxFailed(err)) {
            fprintf(stderr, "RT Loop AI Read Failed: %d\n", err);
            break;
        }

        /* 4. Update Filter */
        float64 y_hat = filter_update(&state_estimator, data_read);

        /* 5. Log state to pre-allocated RAM */
        record_buffer[sample_ctr].k = sample_ctr;
        record_buffer[sample_ctr].u = data_write;
        record_buffer[sample_ctr].y_raw = data_read;
        record_buffer[sample_ctr].y_hat = y_hat;
    }

    return NULL;
}

int main(void)
{
    struct sched_param param;
    pthread_attr_t attr;
    pthread_t thread;

    /* Initialize hardware and evaluate return constraint */
    if (init_board() < 0) {
        fprintf(stderr, "Hardware initialization failed. Aborting.\n");
        return EXIT_FAILURE;
    }

    /* Lock process memory to prevent page faults */
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("mlockall failed");
        goto cleanup;
    }

    /* Configure real-time thread attributes */
    if (pthread_attr_init(&attr)) {
        perror("pthread_attr_init failed");
        goto cleanup;
    }

    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);

    param.sched_priority = RT_PRIORITY;
    pthread_attr_setschedparam(&attr, &param);

    /* Execute real-time task */
    if (pthread_create(&thread, &attr, control_loop_task, NULL)) {
        perror("pthread_create failed");
        goto cleanup;
    }

    pthread_join(thread, NULL);

    /* Post-execution disk I/O dump */
    FILE * exp_data_f = fopen("experimento.txt", "w");
    if (exp_data_f) {
        fprintf(exp_data_f, "k, u, y_raw, y_hat\n");
        for (int i = 0; i < N_SAMPLES; ++i) {
            /* Only write valid samples if loop terminated early */
            if (i > 0 && record_buffer[i].k == 0 && record_buffer[i].u == 0.0)
                break;
            fprintf(exp_data_f, "%d, %lf, %lf, %lf\n", record_buffer[i].k, record_buffer[i].u,
                    record_buffer[i].y_raw, record_buffer[i].y_hat);
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
