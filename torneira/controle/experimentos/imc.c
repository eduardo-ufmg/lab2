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
#define N_SAMPLES 9000 /* 15 minutes at 10 Hz */
#define WAIT_TIMEOUT_SAMPLES 3000 /* 5 minutes maximum wait time */
#define RT_PRIORITY 90

/* System Constants */
#define OP_U 8.0
#define OP_Y 23.0
#define SENSOR_GAIN 10.0

/* Filter Parameters */
#define MEDIAN_SAMPLES 5
#define ALPHA 0.88

/* Controller Parameters */
#define CTRL_NUM_1 -0.007772
#define CTRL_NUM_2 0.01459
#define CTRL_NUM_3 -0.006819

#define CTRL_DEN_1 2.869
#define CTRL_DEN_2 -2.739
#define CTRL_DEN_3 0.8702

#define DAQmxErrChk(functionCall)                                \
  do {                                                           \
    int32 error = (functionCall);                                \
    if (DAQmxFailed(error)) {                                    \
      char errBuff[2048] = { '\0' };                             \
      DAQmxGetExtendedErrorInfo(errBuff, 2048);                  \
      fprintf(stderr, "DAQmx Error [%d]: %s\n", error, errBuff); \
      goto Error;                                                \
    }                                                            \
  } while (0)

typedef enum { SYSTEM_STATE_WAIT_OP, SYSTEM_STATE_CONTROL } system_state_t;

typedef struct
{
  float64 ref;
  float64 y_hat;
  float64 u;
} sample_t;

typedef struct
{
  float64 y_hist[MEDIAN_SAMPLES];
  float64 y_hat_prev;
} filter_state_t;

typedef struct
{
  float64 u_hist[3];
  float64 e_hist[3];
} controller_state_t;

static inline void filter_init(filter_state_t *f)
{
  memset(f->y_hist, 0, sizeof(f->y_hist));
  f->y_hat_prev = 0.0;
}

static inline float64 filter_update(filter_state_t *f, float64 y_n)
{
  float64 temp_hist[MEDIAN_SAMPLES];

  for (int i = MEDIAN_SAMPLES - 1; i > 0; --i) { f->y_hist[i] = f->y_hist[i - 1]; }
  f->y_hist[0] = y_n;

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
  float64 y_hat_volts = ALPHA * f->y_hat_prev + (1.0 - ALPHA) * x_n;

  f->y_hat_prev = y_hat_volts;

  /* Return strictly in voltage domain */
  return y_hat_volts;
}

static inline void controller_init(controller_state_t *c, float64 u0, float64 e0)
{
  for (int i = 0; i < 3; ++i) {
    c->u_hist[i] = u0;
    c->e_hist[i] = e0;
  }
}

static inline float64 controller_update_euler(controller_state_t *c, float64 e_n)
{
  float64 u_k = CTRL_DEN_1 * c->u_hist[0] + CTRL_DEN_2 * c->u_hist[1] + CTRL_DEN_3 * c->u_hist[2]
                + CTRL_NUM_1 * c->e_hist[0] + CTRL_NUM_2 * c->e_hist[1] + CTRL_NUM_3 * c->e_hist[2];

  c->u_hist[2] = c->u_hist[1];
  c->u_hist[1] = c->u_hist[0];
  c->u_hist[0] = u_k;

  c->e_hist[2] = c->e_hist[1];
  c->e_hist[1] = c->e_hist[0];
  c->e_hist[0] = e_n;

  return u_k;
}

/* Global state */
TaskHandle AItaskHandle = 0;
TaskHandle AOtaskHandle = 0;
sample_t record_buffer[N_SAMPLES];
volatile int32 rt_error_code = 0;
volatile int samples_recorded = 0;
volatile int timeout_flag = 0;

static int init_board(void)
{
  DAQmxErrChk(DAQmxCreateTask("AI_Task", &AItaskHandle));
  DAQmxErrChk(DAQmxCreateTask("AO_Task", &AOtaskHandle));

  DAQmxErrChk(
    DAQmxCreateAIVoltageChan(AItaskHandle, "Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, NULL));
  DAQmxErrChk(DAQmxCreateAOVoltageChan(AOtaskHandle, "Dev1/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, NULL));

  DAQmxErrChk(
    DAQmxCfgSampClkTiming(AItaskHandle, "", SAMPLES_PER_SECOND, DAQmx_Val_Rising, DAQmx_Val_HWTimedSinglePoint, 1));
  DAQmxErrChk(DAQmxCfgSampClkTiming(
    AOtaskHandle, "ai/SampleClock", SAMPLES_PER_SECOND, DAQmx_Val_Rising, DAQmx_Val_HWTimedSinglePoint, 1));

  DAQmxErrChk(DAQmxSetRealTimeWaitForNextSampClkWaitMode(AItaskHandle, DAQmx_Val_WaitForInterrupt));
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

void *control_loop_task(void *arg)
{
  (void)arg;
  float64 data_read = 0.0;
  float64 data_write = OP_U; /* Initialize output to OP immediately */
  int32 err = 0;

  filter_state_t state_estimator;
  filter_init(&state_estimator);

  controller_state_t controller;

  system_state_t current_state = SYSTEM_STATE_WAIT_OP;
  unsigned op_counter = 0;
  unsigned wait_timeout_counter = 0;

  while (1) {
    /* 1. Write DAC (Buffered for next edge) */
    err = DAQmxWriteAnalogScalarF64(AOtaskHandle, 1, 10.0, data_write, NULL);
    if (DAQmxFailed(err)) {
      rt_error_code = err;
      break;
    }

    /* 2. Wait for HW Edge */
    err = DAQmxWaitForNextSampleClock(AItaskHandle, 10.0, NULL);
    if (DAQmxFailed(err)) {
      rt_error_code = err;
      break;
    }

    /* 3. Read ADC */
    err = DAQmxReadAnalogScalarF64(AItaskHandle, 10.0, &data_read, NULL);
    if (DAQmxFailed(err)) {
      rt_error_code = err;
      break;
    }

    /* 4. Filter and Scale */
    float64 y_hat_volts = filter_update(&state_estimator, data_read);
    float64 y_hat_celsius = y_hat_volts * SENSOR_GAIN;

    /* 5. State Machine Evaluation */
    if (current_state == SYSTEM_STATE_WAIT_OP) {

      data_write = OP_U;

      if (fabs(y_hat_celsius - OP_Y) < 0.5) {
        op_counter++;
      } else {
        op_counter = 0;
      }

      wait_timeout_counter++;
      if (wait_timeout_counter >= WAIT_TIMEOUT_SAMPLES) {
        timeout_flag = 1;
        break;
      }

      if (op_counter >= 50) {
        /* Bumpless transfer initialization */
        controller_init(&controller, OP_U, 0.0);
        current_state = SYSTEM_STATE_CONTROL;
      }

    } else if (current_state == SYSTEM_STATE_CONTROL) {

      float64 ref = 0.0;
      if (samples_recorded > 3600) {
        ref = 25.0;
      } else if (samples_recorded > 1800) {
        ref = 30.0;
      } else {
        ref = 23.0;
      }

      float64 error = ref - y_hat_celsius;
      data_write = controller_update_euler(&controller, error);

      /* Log state */
      record_buffer[samples_recorded].ref = ref;
      record_buffer[samples_recorded].y_hat = y_hat_celsius;
      record_buffer[samples_recorded].u = data_write;

      samples_recorded++;

      if (samples_recorded >= N_SAMPLES) { break; /* Target completed */ }
    }
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

  if (timeout_flag) { fprintf(stderr, "Experiment aborted: Operating point not reached within timeout period.\n"); }

  if (rt_error_code != 0) {
    char errBuff[2048] = { '\0' };
    DAQmxGetExtendedErrorInfo(errBuff, 2048);
    fprintf(stderr, "Real-time task aborted with DAQmx Error [%d]: %s\n", rt_error_code, errBuff);
  }

  FILE *exp_data_f = fopen("experimento_imc.txt", "w");
  if (exp_data_f) {
    fprintf(exp_data_f, "ref, y_hat, u\n");
    for (int i = 0; i < samples_recorded; ++i) {
      fprintf(exp_data_f, "%lf, %lf, %lf\n", record_buffer[i].ref, record_buffer[i].y_hat, record_buffer[i].u);
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

  return (rt_error_code == 0 && timeout_flag == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
