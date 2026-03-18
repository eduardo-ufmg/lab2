#define _GNU_SOURCE

#include <NIDAQmx.h>
#include <errno.h>
#include <gsl/gsl_statistics_double.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

#define NSEC_TO_SEC 1000000000
#define SAMPLING_TIME (100 /*ms*/ * 1000000) //us
#define N_SAMPLES 5400
#define SAMPLES_PER_SECOND (1.0 * NSEC_TO_SEC / SAMPLING_TIME)
#define RT_PRIORITY 90 // between 1 and 99, where 99 is the highest priority

#define DAQmxErrChk(functionCall)                                                                  \
    do {                                                                                           \
        if (DAQmxFailed(error = (functionCall)))                                                   \
            goto Error;                                                                            \
    } while (0)

int32 error = 0;
char errBuff[2048] = {'\0'};
TaskHandle AItaskHandle = 0;
TaskHandle AOtaskHandle = 0;

struct period_info
{
    struct timespec next_period;
    long long period_ns;
};

static void inc_period(struct period_info * pinfo)
{
    pinfo->next_period.tv_nsec += pinfo->period_ns;

    while (pinfo->next_period.tv_nsec >= NSEC_TO_SEC) {
        pinfo->next_period.tv_sec++;
        pinfo->next_period.tv_nsec -= NSEC_TO_SEC;
    }
}

static void periodic_task_init(struct period_info * pinfo)
{
    pinfo->period_ns = SAMPLING_TIME;

    clock_gettime(CLOCK_MONOTONIC, &(pinfo->next_period));
}

static void write_dac(float64 data)
{
    DAQmxErrChk(DAQmxWriteAnalogScalarF64(AOtaskHandle, 1, 0.0, data, NULL));

    return;
Error:
    if (DAQmxFailed(error)) {
        DAQmxGetExtendedErrorInfo(errBuff, 2048);
        printf("DAQmx ErrorRead: %s\n", errBuff);
        return;
    }
}

static void read_adc(float64 * data)
{
    DAQmxErrChk(DAQmxReadAnalogScalarF64(AItaskHandle, 10.0, data, NULL));

    return;
Error:
    if (DAQmxFailed(error)) {
        DAQmxGetExtendedErrorInfo(errBuff, 2048);
        printf("DAQmx ErrorRead: %s\n", errBuff);
        return;
    }
}

static void wait_rest_of_period(struct period_info * pinfo)
{
    int ret;

    inc_period(pinfo);

    do {
        ret = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &pinfo->next_period, NULL);
    } while (ret == EINTR);

    if (ret != 0) {
        perror("clock_nanosleep");
    }
}

void * simple_cyclic_task(void * data)
{
    (void)data;
    int sample_ctr = 0;
    struct period_info pinfo;
    float64 data_read;
    float64 data_write = 8;

    periodic_task_init(&pinfo);

    FILE * exp_data_f = fopen("experimento.txt", "w");
    fprintf(exp_data_f, "k, u, y\n");

    while (sample_ctr < N_SAMPLES) {

        if (sample_ctr > 3600) {
            data_write = 8;
        } else if (sample_ctr > 1800) {
            data_write = 2;
        }

        read_adc(&data_read);
        write_dac(data_write);

        fprintf(exp_data_f, "%d, %lf, %lf\n", sample_ctr, data_write, data_read);
        fflush(exp_data_f);

        printf("%d, %lf, %lf\n", sample_ctr, data_write, data_read);

        wait_rest_of_period(&pinfo);
        sample_ctr++;
    }

    fclose(exp_data_f);

    return NULL;
}

static int init_board(void)
{

    DAQmxErrChk(DAQmxCreateTask("", &AItaskHandle));
    DAQmxErrChk(DAQmxCreateTask("", &AOtaskHandle));

    DAQmxErrChk(DAQmxCreateAIVoltageChan(AItaskHandle, "Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0,
                                         10.0, DAQmx_Val_Volts, NULL));
    DAQmxErrChk(
        DAQmxCreateAOVoltageChan(AOtaskHandle, "Dev1/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, NULL));

    DAQmxErrChk(DAQmxCfgSampClkTiming(AItaskHandle, "", SAMPLES_PER_SECOND, DAQmx_Val_Rising,
                                      DAQmx_Val_HWTimedSinglePoint, 1));
    DAQmxErrChk(DAQmxCfgSampClkTiming(AOtaskHandle, "", SAMPLES_PER_SECOND, DAQmx_Val_Rising,
                                      DAQmx_Val_HWTimedSinglePoint, 1));

    DAQmxSetRealTimeWaitForNextSampClkWaitMode(AItaskHandle, DAQmx_Val_WaitForInterrupt);
    DAQmxSetRealTimeConvLateErrorsToWarnings(AItaskHandle, 1);
    DAQmxSetReadWaitMode(AItaskHandle, DAQmx_Val_Poll);

    DAQmxErrChk(DAQmxStartTask(AItaskHandle));
    DAQmxErrChk(DAQmxStartTask(AOtaskHandle));

    return 0;

Error:
    if (DAQmxFailed(error)) {
        DAQmxGetExtendedErrorInfo(errBuff, 2048);
        printf("DAQmx Error: %s\n", errBuff);
    }
    if (AItaskHandle != 0) {
        DAQmxStopTask(AItaskHandle);
        DAQmxClearTask(AItaskHandle);
    }

    return -1;
}

int main()
{
    struct sched_param param;
    pthread_attr_t attr;
    pthread_t thread;
    int ret;

    struct rlimit rlim;

    init_board();

    getrlimit(RLIMIT_RTTIME, &rlim);
    rlim.rlim_cur = (10 * 60 * 1000000); //us
    setrlimit(RLIMIT_RTTIME, &rlim);

    /* Lock memory */
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        printf("mlockall failed: %m\n");
        exit(-2);
    }

    /* Initialize pthread attributes (default values) */
    ret = pthread_attr_init(&attr);
    if (ret) {
        printf("init pthread attributes failed\n");
        goto out;
    }

    /* Set a specific stack size  */
    ret = pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN);
    if (ret) {
        printf("pthread setstacksize failed\n");
        goto out;
    }

    /* Use scheduling parameters of attr */
    ret = pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    if (ret) {
        printf("pthread setinheritsched failed\n");
        goto out;
    }

    /* Set scheduler policy and priority of pthread */
    ret = pthread_attr_setschedpolicy(&attr, SCHED_RR);
    if (ret) {
        printf("pthread setschedpolicy failed\n");
        goto out;
    }

    pthread_attr_getschedparam(&attr, &param);
    param.sched_priority = RT_PRIORITY;
    ret = pthread_attr_setschedparam(&attr, &param);
    if (ret) {
        printf("pthread setschedparam failed\n");
        goto out;
    }

    /* Create a pthread with specified attributes */
    ret = pthread_create(&thread, &attr, simple_cyclic_task, NULL);
    if (ret) {
        printf("create pthread failed\n");
        goto out;
    }

    /* Join the thread and wait until it is done */
    ret = pthread_join(thread, NULL);
    if (ret) {
        printf("join pthread failed: %m\n");
    }

    //Stop and clear task
    DAQmxStopTask(AItaskHandle);
    DAQmxClearTask(AItaskHandle);
    DAQmxStopTask(AOtaskHandle);
    DAQmxClearTask(AOtaskHandle);

out:
    return ret;
}
