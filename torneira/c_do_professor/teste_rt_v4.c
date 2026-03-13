/*
 * Programa de Teste do Desempenho de Tempo Real com uso do Driver NIDAQmx
 * 
 */
// Para compilar: (PC Modelagem)
//gcc teste_rt_v2.c /usr/lib/x86_64-linux-gnu/libnidaqmx.so.23.8.0 -pthread -lm -lgsl -lgslcblas -o teste_rt
//Executar como superusuário

// Para acompanhar os processos e suas prioridades pelo terminal:
// ps -eLo pid,class,rtprio,ni,comm | grep -i "pid_app"
// Para acompanhar interrupções relevantes (nipalk para o driver,rtc0 para o relógio):
// ps -eLo pid,class,rtprio,ni,comm | grep -i "irq"
// Para mudar prioridade das interrupções:
// chrt -f -p 90 pid #onde pid é o process id 

// Aula 2 - Dividir a tarefa de acesso a placa em duas funções, uma para leitura
//          e outra para escrita dos dados. A partir de um tempo de amostragem de 
//          100ms realizar 3 faixas de tensão do sinal de controle:
//               * n_amostras < 1200: 8V
//               * 1200 < n_amostras < 2400 : 2V
//               * 2400 < n_amostrasr < 3600 : 8V


#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <errno.h>
#include <NIDAQmx.h>
#include <sys/resource.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_statistics_double.h>

#define NSEC_TO_SEC 1000000000
#define SAMPLING_TIME 100000000        // Ts = 100ms
#define AMOSTRAS 3600                   // 3600 amostras -> 6 min de ensaio
#define SAMPLES_PER_SECOND (1.0*NSEC_TO_SEC/SAMPLING_TIME)
#define RT_PRIORITY 90

#define DAQmxErrChk(functionCall) if( DAQmxFailed(error=(functionCall)) ) goto Error; else

//Variáveis da placa de aquisição
int32       error=0;
char        errBuff[2048]={'\0'};
TaskHandle  AItaskHandle=0;
TaskHandle  AOtaskHandle=0;

int n[AMOSTRAS];
float64 sinalDeControle[AMOSTRAS];
float64 sinalDeResposta[AMOSTRAS];
float64 yf[AMOSTRAS];
float64 ef[AMOSTRAS];
float64 r[AMOSTRAS];
float64 e[AMOSTRAS];

unsigned int count=0;


struct period_info {
        struct timespec next_period;
        long period_ns;
};

static void filtered_signal(int k){
        ef[k] = abs( (yf[k]-sinalDeResposta[k]) / yf[k] );

        if(ef[k]>0.1){
                sinalDeResposta[k] = yf[k-1];
        }
        else{
                yf[k] = 0.9775*yf[k-1] + 0.01124*sinalDeResposta[k-1] + 0.01124*sinalDeResposta[k];
        }
}

static void controller(int k){
        sinalDeControle[k] = -1.514*e[k] + 1.507*e[k-1] + sinalDeControle[k-1];

        if( sinalDeControle[k] > 10) {
                sinalDeControle[k] = 10;
        }
        else if ( sinalDeControle[k] < 0) {
                sinalDeControle[k] = 0;
        }
}

static void inc_period(struct period_info *pinfo) {
        pinfo->next_period.tv_nsec += pinfo->period_ns;


        while (pinfo->next_period.tv_nsec >= 1000000000) {
                /* timespec nsec overflow */
                pinfo->next_period.tv_sec++;
                pinfo->next_period.tv_nsec -= 1000000000;
        }
}               

static void periodic_task_init(struct period_info *pinfo) {
        /* for simplicity, hardcoding a 1ms period */
        pinfo->period_ns = SAMPLING_TIME;

        clock_gettime(CLOCK_MONOTONIC, &(pinfo->next_period));
}

static float64 do_rt_task_read() {
        /* Do RT stuff here. */
        int i;
        float64 data;

	DAQmxErrChk(DAQmxReadAnalogScalarF64(AItaskHandle, 10.0,&data,NULL));
Out:
	return data*10;	// Conversão para temperatura
Error:
       if( DAQmxFailed(error) )
                DAQmxGetExtendedErrorInfo(errBuff,2048);
       if( DAQmxFailed(error) )
                printf("DAQmx ErrorRead: %s\n",errBuff);
                return -1;
}

static void do_rt_task_write(float64 data) {
	int i;
		 		
	DAQmxErrChk(DAQmxWriteAnalogScalarF64(AOtaskHandle, 1,0.0,data,NULL));
				
Out:
	return;
Error:
	if ( DAQmxFailed(error) )
		DAQmxGetExtendedErrorInfo(errBuff,2048);
	if ( DAQmxFailed(error) )
	        printf("DAQmx ErrorRead: %s\n",errBuff);
		return;
}

static void wait_rest_of_period(struct period_info *pinfo) {
	int islate;

        DAQmxWaitForNextSampleClock(AItaskHandle,15.0,&islate);
}

void *simple_cyclic_task(void *data) {
        struct period_info pinfo;
        struct timespec tempo;
        struct timespec tempo2;
        int j;
        struct rlimit rlim;
        double media=0;
        double momento=0;
        double AvgLatency=0;
        int ret;
        float64 u;
        float64 y;
          
	ret=sched_getscheduler(0);
	printf("priority=%d\n",ret);

        FILE* ArqTest;
        ArqTest = fopen( "Dados.txt", "w" );
        if( ArqTest == NULL ){
                printf("\n  Erro ao Criar Arquivo Tempo");
                //printf("\n  Significado: %s \n", strerror( errno));
        }

        periodic_task_init(&pinfo);
        while (count<AMOSTRAS) {
                //ret=clock_gettime(CLOCK_MONOTONIC, &tempo);
                
                y = do_rt_task_read();
                y=30.5;
                sinalDeResposta[count] = y;

                if(count == 0){
                        yf[count] = y;
                }
                else{
                        filtered_signal(count);
                }

                // Definição da referência

                if(count < 1200){
                        r[count] = 30;
                }
                else if(count >= 1200 && count < 2400) {
                        r[count] = 32;
                }
                else{
                        r[count] = 30;
                }

                e[count] = r[count] - yf[count];

                // Sinal de controle
        		if (count == 0) {
        			sinalDeControle[count] = 8;
        		} else {
        			controller(count);
        		}
                
                //if(count < 1200){
                //        u = 8;
                //}
                //else if(count >= 1200 && count < 2400) {
                //        u = 2;
                //}
                //else{
                //        u = 8;
                //}

                u = sinalDeControle[count];
                do_rt_task_write(u);

                n[count] = count;

                if ( count % 10 == 0 ) {
                        printf("K: %d ; MV: %f, PV: %f, r: %f, e: %f\n", count, u, yf[count], r[count], e[count]);
                }
                
                wait_rest_of_period(&pinfo);
                count++;
        }
        for(j=0;j<AMOSTRAS;j++){
                fprintf(ArqTest,"%d,%f,%f,%f\n", n[j], sinalDeControle[j], yf[j], r[j]);
                //fflush(ArqTest);
        }
        fclose(ArqTest);

        return NULL;
}

//Função para inicializar driver da placa de aquisição
static int Inicializa_placa(void) {
	
	DAQmxErrChk (DAQmxCreateTask("", &AItaskHandle)); //cria tarefa de entrada analógica
	DAQmxErrChk (DAQmxCreateTask("", &AOtaskHandle)); //cria tarefa de saída analógica

    //configura canal de entrada
	DAQmxErrChk(DAQmxCreateAIVoltageChan(AItaskHandle, "Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, NULL));
    //configura canal de saída
	DAQmxErrChk(DAQmxCreateAOVoltageChan(AOtaskHandle, "Dev1/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, NULL));

    //configura timers para tarefas de leitura e escrita
	DAQmxErrChk(DAQmxCfgSampClkTiming(AItaskHandle, "", SAMPLES_PER_SECOND, DAQmx_Val_Rising,DAQmx_Val_HWTimedSinglePoint,1));//DAQmx_Val_ContSamps or DAQmx_Val_HWTimedSinglePoint
	DAQmxErrChk(DAQmxCfgSampClkTiming(AOtaskHandle, "",SAMPLES_PER_SECOND, DAQmx_Val_Rising,DAQmx_Val_HWTimedSinglePoint,1));//DAQmx_Val_HWTimedSinglePoint or DAQmx_Val_ContSamps
	

    //permitirá que a thread de tempo real durma enquanto aguarda o próximo interrupt
    DAQmxSetRealTimeWaitForNextSampClkWaitMode(AItaskHandle,DAQmx_Val_WaitForInterrupt); 
    DAQmxSetRealTimeConvLateErrorsToWarnings(AItaskHandle,1);
    //evita que tarefa durma enquanto aguarda a leitura, o que poderia ceder o controle para um processo de menor prioridade
    DAQmxSetReadWaitMode(AItaskHandle,DAQmx_Val_Poll);

	// Inicia tarefas
	DAQmxErrChk(DAQmxStartTask(AItaskHandle));
	DAQmxErrChk(DAQmxStartTask(AOtaskHandle));

out:
        return 0;

Error:
       if( DAQmxFailed(error) )
             DAQmxGetExtendedErrorInfo(errBuff,2048);
       if( AItaskHandle!=0 )  {
              DAQmxStopTask(AItaskHandle);
              DAQmxClearTask(AItaskHandle);
       }
       if( DAQmxFailed(error) )
              printf("DAQmx Error: %s\n",errBuff);
       return 0;
	
}

int main() {
    struct sched_param param;
    pthread_attr_t attr,attr2;
    pthread_t thread;
    int ret;
       
	struct rlimit rlim;

	Inicializa_placa();
	
	//Define tempo máximo que um processo de tempo real pode ocupar o processador
    //Evita que o programa congele o computador
	getrlimit(RLIMIT_RTTIME,&rlim);
	rlim.rlim_cur=600000000; // tempo limite 10 min -> 600s
	setrlimit(RLIMIT_RTTIME,&rlim);
	
        /* Lock memory */
        if(mlockall(MCL_CURRENT|MCL_FUTURE) == -1) {
                printf("mlockall failed: %m\n");
                exit(-2);
        }
      
        /* Initialize pthread attributes (default values) */
        ret = pthread_attr_init(&attr);
        if (ret) {
                printf("init pthread attributes failed\n");
                goto out;
        }
        
        ret = pthread_attr_init(&attr2);
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

	ret = pthread_attr_setstacksize(&attr2, PTHREAD_STACK_MIN);
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
	ret = pthread_attr_setinheritsched(&attr2, PTHREAD_EXPLICIT_SCHED);
        if (ret) {
                printf("pthread setinheritsched failed\n");
                goto out;
        }


        /* Set scheduler policy and priority of pthread */
        ret = pthread_attr_setschedpolicy(&attr, SCHED_RR);
        //  ret = pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
        if (ret) {
                printf("pthread setschedpolicy failed\n");
                goto out;
        }
	ret = pthread_attr_setschedpolicy(&attr2, SCHED_RR);
        //  ret = pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
        if (ret) {
                printf("pthread setschedpolicy failed\n");
                goto out;
        }

        pthread_attr_getschedparam(&attr,&param);
        param.sched_priority = RT_PRIORITY;
        ret = pthread_attr_setschedparam(&attr, &param);
        if (ret) {
                printf("pthread setschedparam failed\n");
                goto out;
        }
        param.sched_priority = 1;
	ret = pthread_attr_setschedparam(&attr2, &param);
        if (ret) {
                printf("pthread setschedparam failed\n");
                goto out;
        }

        /* Create a pthread with specified attributes */
        ret = pthread_create(&thread, &attr, simple_cyclic_task, NULL);
        //    ret = pthread_create(&thread,NULL, simple_cyclic_task, NULL);  //NO REAL TIME
        if (ret) {
                printf("create pthread failed\n");
                goto out;
        }


        /* Join the thread and wait until it is done */
        ret = pthread_join(thread, NULL);
        if (ret)
                printf("join pthread failed: %m\n");

        //Stop and clear task
        DAQmxStopTask(AItaskHandle);
        DAQmxClearTask(AItaskHandle);
        DAQmxStopTask(AOtaskHandle);
        DAQmxClearTask(AOtaskHandle);
        
out:
        return ret;

Error:
       if( DAQmxFailed(error) )
             DAQmxGetExtendedErrorInfo(errBuff,2048);
       if( AItaskHandle!=0 )  {
              DAQmxStopTask(AItaskHandle);
              DAQmxClearTask(AItaskHandle);
       }
       if( DAQmxFailed(error) )
              printf("DAQmx Error: %s\n",errBuff);
              return 0;

}
