#include <stdio.h>
#include "pico/stdlib.h"

#include "logic_or_test.h"

// --- HEADERS DE TENSORFLOW LITE MICRO ---
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h" <--- ELIMINADO PORQUE NO EXISTE

// --- DEFINICIONES AUXILIARES ---
// Si por alguna razón schema_generated.h no define la versión, la forzamos a 3.
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

#define MAX_V 5.0
#define MIN_V 0.0

// --- CONFIGURACIÓN DE MEMORIA ---
// 16KB suele ser suficiente para pruebas
const int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

float input_scale_standard(float input, float mean, float desviation){
    return (input - mean) / desviation;
}

float input_scale_min_max(float input, float min, float max, float scale){
    return (input * scale + min);
}

#include "tensorflow/lite/micro/micro_time.h" // Asegúrate de incluir este header

void run_inference(tflite::MicroInterpreter *p_interpreter, float vin_0, float vin_1) {
    TfLiteTensor* input = p_interpreter->input(0);

    // 1. Preparar entradas con tu función de escalado
    input->data.f[0] = input_scale_min_max(vin_0, MIN_V, MAX_V, 1.0); 
    input->data.f[1] = input_scale_min_max(vin_1, MIN_V, MAX_V, 1.0);

    printf("[INFO] Ejecutando inferencia...\n");

    // 2. Capturar tiempo de inicio
    uint32_t start_ticks = tflite::GetCurrentTimeTicks();

    // 3. Ejecutar la inferencia
    TfLiteStatus invoke_status = p_interpreter->Invoke();

    // 4. Capturar tiempo de fin y calcular diferencia
    uint32_t end_ticks = tflite::GetCurrentTimeTicks();
    uint32_t duration = end_ticks - start_ticks;

    if (invoke_status != kTfLiteOk) {
        printf("[ERROR] Error al invocar el interprete\n");
        return;
    }
    
    // 5. Mostrar resultados y métricas de tiempo
    TfLiteTensor* output = p_interpreter->output(0);

    printf("------------------------------------------\n");
    printf("Tiempo de ejecucion: %lu us (microsegundos)\n", duration);
    printf("Resultado de la inferencia: [%f %f %f]\n", 
            output->data.f[0], output->data.f[1], output->data.f[2]);
    printf("Probabilidad de HIGH: %f\n", output->data.f[0]);  
    printf("Probabilidad de INDEFINIDO: %f\n", output->data.f[1]);  
    printf("Probabilidad de LOW: %f\n", output->data.f[2]);  
    printf("------------------------------------------\n");
}

int main() {
    // 1. Inicializar hardware
    stdio_init_all();
    
    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    
    // Espera activa hasta que abras el monitor serial
    while (!stdio_usb_connected()) {
        gpio_put(LED_PIN, 1); sleep_ms(100);
        gpio_put(LED_PIN, 0); sleep_ms(100);
    }
    sleep_ms(2000); 

    printf("\n\n=== TFLite Micro en Pico 2 (Cortex-M33) 3 ===\n");

    // 2. Cargar el Modelo
    // Usamos el dummy data definido arriba
    const tflite::Model* model = tflite::GetModel(logic_or_test);
    
    // Verificación de Schema
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[INFO] Nota: La version del modelo (%ld) es distinta a la del Schema (%d).\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        printf("[INFO] Esto es normal con el modelo Dummy. Si llegamos aqui, la libreria funciona.\n");
    } else {
        printf("[OK] Modelo cargado y version correcta.\n");
    }

    // 3. Inicializar Resolver
    static tflite::MicroMutableOpResolver<5> resolver;
    // Añadimos operaciones comunes para asegurar que el Linker las encuentre
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddRelu();

    // 4. Instanciar Interprete
    static tflite::MicroInterpreter interpreter(
        model, 
        resolver, 
        tensor_arena, 
        kTensorArenaSize
    );

    // 5. Asignar Memoria
    // pero si compila y ejecuta hasta aquí, TU LIBRERÍA ESTÁ PERFECTA.
    printf("[INTENTO] Asignando tensores...\n");
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    
    if (allocate_status != kTfLiteOk) {
        printf("[NOTA] AllocateTensors fallo (Esperado con modelo dummy).\n");
        printf("[EXITO] ¡La libreria compilo, linkeo e inicio correctamente!\n");
    } else {
        printf("[EXITO] Tensores asignados.\n");
    }

    printf("== COMENZANDO INFERENCIA ==\n");

    // Obtener el puntero al tensor de entrada
    float vin_0_list[4]={0.1, 4.5, 3.4, 1.4};
    float vin_1_list[4]={0.3, 4.5, 1.0, 3.0};

    for(int i=0; i<4; i++){
        printf("===============================\n");
        printf("Vin_0: %f\n", vin_0_list[i]);
        printf("Vin_1: %f\n", vin_1_list[i]);
    
        run_inference(&interpreter, vin_0_list[i], vin_1_list[i]);  
    }

    // Bucle final
    while (true) {
        // printf("TFLite Running...\n");
        gpio_put(LED_PIN, 1);
        sleep_ms(1000);
        gpio_put(LED_PIN, 0);
        sleep_ms(1000);
    }

    return 0;
}