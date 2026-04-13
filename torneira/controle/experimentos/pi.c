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
#define N_SAMPLES 1800 /* 3 minutes at 10 Hz */
#define WAIT_TIMEOUT_SAMPLES 3000 /* 5 minutes maximum wait time */
#define RT_PRIORITY 90

/* System Constants */
#define OP_U 8.0
#define OP_Y 26.0
#define SENSOR_GAIN 10.0

/* Filter Parameters */
#define MEDIAN_SAMPLES 5
#define ALPHA 0.88

/* Controller Parameters */
#define Kp -0.663
#define Ti +12.50

#define Ts 0.1

#define U_MIN 2.0
#define U_MAX 8.0

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
  float64 u_prev;
  float64 e_prev;
} controller_state_t;

static inline void filter_init(filter_state_t *f, float64 y0)
{
  memset(f->y_hist, 0, sizeof(f->y_hist));
  f->y_hat_prev = y0;
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
  c->u_prev = u0;
  c->e_prev = e0;
}

static inline float64 controller_update_euler(controller_state_t *c, float64 e_n)
{
  /* Backward Euler coefficients */
  float64 q0 = Kp * (1.0 + Ts / Ti);
  float64 q1 = -Kp;

  /* Compute raw control action */
  float64 u_n = c->u_prev + q0 * e_n + q1 * c->e_prev;

  /* Anti-windup via output saturation */
  if (u_n > U_MAX) {
    u_n = U_MAX;
  } else if (u_n < U_MIN) {
    u_n = U_MIN;
  }

  /* State update */
  c->u_prev = u_n;
  c->e_prev = e_n;

  return u_n;
}

static inline float64 controller_update_tustin(controller_state_t *c, float64 e_n)
{
  /* Tustin (Bilinear) coefficients */
  float64 q0 = Kp * (1.0 + Ts / (2.0 * Ti));
  float64 q1 = -Kp * (1.0 - Ts / (2.0 * Ti));

  /* Compute raw control action */
  float64 u_n = c->u_prev + q0 * e_n + q1 * c->e_prev;

  /* Anti-windup via output saturation */
  if (u_n > U_MAX) {
    u_n = U_MAX;
  } else if (u_n < U_MIN) {
    u_n = U_MIN;
  }

  /* State update */
  c->u_prev = u_n;
  c->e_prev = e_n;

  return u_n;
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
  filter_init(&state_estimator, OP_Y / SENSOR_GAIN);

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

      if (fabs(y_hat_celsius - OP_Y) < 1.0) {
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

      if (wait_timeout_counter % 10 == 0) { printf("%lf\n", y_hat_celsius); }

    } else if (current_state == SYSTEM_STATE_CONTROL) {

      float64 ref = 0.0;
      if (samples_recorded > 2 * N_SAMPLES / 3) {
        ref = 28.0;
      } else if (samples_recorded > N_SAMPLES / 3) {
        ref = 31.0;
      } else {
        ref = 26.0;
      }

      float64 error = ref - y_hat_celsius;
      data_write = controller_update_euler(&controller, error);

      /* Log state */
      record_buffer[samples_recorded].ref = ref;
      record_buffer[samples_recorded].y_hat = y_hat_celsius;
      record_buffer[samples_recorded].u = data_write;

      samples_recorded++;

      if (samples_recorded % 10 == 0) { printf("%lf, %lf, %lf\n", ref, y_hat_celsius, data_write); }

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

  FILE *exp_data_f = fopen("experimento_pi.txt", "w");
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
