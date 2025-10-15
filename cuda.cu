#include <iostream>
#include <vector>
#include <cmath>      // Для M_PI, sin, cos, log, abs
#include <iomanip>    // Для std::setprecision, std::fixed
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // Для blockIdx, threadIdx и т.д.


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


__device__ double device_f(double x) {
		// f(x) = cot(x) / (ln(1+sin(x)) * sin(1+sin(x)))
		// cot(x) = cos(x) / sin(x)

		// Метод средних прямоугольников гарантирует, что x не будет равен 0 или M_PI.
		// Однако sin(x) может быть очень близок к 0, если x близок к 0 или M_PI.

		double sin_x = sin(x);
		double cos_x = cos(x);

		// Знаменатель, член 1: log(1+sin(x))
		double term_1_plus_sin_x = 1.0 + sin_x;
		// Так как 0 < x < PI (из-за метода средних), то 0 < sin(x) <= 1.
		// Следовательно, 1 < 1+sin(x) <= 2.
		// log(term_1_plus_sin_x) всегда будет > log(1) = 0.
		// Проблема может возникнуть, если sin_x настолько мал, что 1.0 + sin_x вычисляется как 1.0 из-за точности.
		double log_term = log(term_1_plus_sin_x);

		// Знаменатель, член 2: sin(1+sin(x))
		// Так как 1 < 1+sin(x) <= 2 (радианы, примерно 57.3 до 114.6 градусов),
		// sin не равен нулю в этом диапазоне [sin(1), sin(2)]. Оба значения положительны.
		// sin(1) ~ 0.84, sin(2) ~ 0.91. Так что этот член безопасен.
		double sin_term_denom = sin(term_1_plus_sin_x);

		// Основной источник INF/NaN будет от cot(x), если sin(x) очень мал,
		// или от log_term, если 1+sin_x численно равно 1.
		// Стандартные математические функции должны корректно обработать это (вернуть INF/NaN).

		double cot_x = cos_x / sin_x; // Это будет INF, если sin_x равен 0.
		
		// Если log_term или sin_term_denom равны нулю (маловероятно для sin_term_denom, возможно для log_term из-за точности)
		// или sin_x равен нулю, то произойдет деление на ноль.
		// CUDA должна обработать это как +/- INF или NaN.
		if (log_term == 0.0 || sin_term_denom == 0.0 || sin_x == 0.0) {
				// Эта ситуация указывает на расходимость или проблемы с численной устойчивостью вблизи сингулярностей.
				// Возвращаем INF со знаком cot_x, если это возможно определить, или просто позволяем произойти делению на ноль.
				if (sin_x == 0.0) return copysign(1.0/0.0, cos_x); // +/- INF для cot(x)
				if (log_term == 0.0) return copysign(1.0/0.0, cot_x); // cot(x) / (0 * non_zero)
				// Другие случаи также приведут к INF/NaN при делении.
		}

		return cot_x / (log_term * sin_term_denom);
}


__global__ void calculate_block_sums_kernel(double a, double b, int n_steps, double* block_sums) {
		// Разделяемая память для суммирования внутри блока
		extern __shared__ double sdata[];

		unsigned int tid_in_block = threadIdx.x; // ID потока внутри блока
		unsigned int block_id = blockIdx.x;       // ID блока
		unsigned int threads_per_block_dim = blockDim.x; // Количество потоков в блоке
		unsigned int total_grid_threads = gridDim.x * blockDim.x; // Общее количество потоков в сетке

		double dx = (b - a) / n_steps;
		double thread_local_sum = 0.0;

		// Цикл с шагом по сетке (grid-stride loop)
		// Каждый поток суммирует f(x_i) для назначенных ему подинтервалов
		for (int i = block_id * threads_per_block_dim + tid_in_block; i < n_steps; i += total_grid_threads) {
				double x_mid = a + (i + 0.5) * dx; // Средняя точка i-го прямоугольника
				
				// Теоретически, x_mid всегда будет > a и < b.
				// Добавим проверку на всякий случай для устойчивости.
				if (x_mid > a && x_mid < b) { 
						 thread_local_sum += device_f(x_mid);
				}
		}

		// Загружаем локальную сумму потока в разделяемую память
		sdata[tid_in_block] = thread_local_sum;
		__syncthreads(); // Синхронизация потоков внутри блока

		// Выполняем редукцию (суммирование) в разделяемой памяти
		for (unsigned int s = threads_per_block_dim / 2; s > 0; s >>= 1) {
				if (tid_in_block < s) {
						sdata[tid_in_block] += sdata[tid_in_block + s];
				}
				__syncthreads(); // Синхронизация после каждого шага редукции
		}

		// Поток с ID 0 каждого блока записывает сумму своего блока в глобальную память
		if (tid_in_block == 0) {
				block_sums[block_id] = sdata[0];
		}
}


void checkCudaError(cudaError_t error, const char* message) {
		if (error != cudaSuccess) {
				std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(error) << ")" << std::endl;
				exit(EXIT_FAILURE);
		}
}


int main() {
		double a = 0.0;
		double b = M_PI - 0.1;
		int n_steps = 100000000;

		std::cout << "Интегрирование f(x) = cot(x) / (ln(1+sin(x)) * sin(1+sin(x)))" << std::endl;
		std::cout << "Метод: средних прямоугольников (сумма f(x_i) * dx, где x_i - середина интервала)" << std::endl;
		std::cout << "Интервал: [" << std::fixed << std::setprecision(1) << a << ", " << b << "]" << std::endl;
		std::cout << "Количество шагов (N): " << n_steps << std::endl;
		
		if (n_steps <= 0) {
				std::cerr << "Количество шагов должно быть положительным." << std::endl;
				return 1;
		}
		
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start); // Запись начала

		double dx = (b - a) / n_steps;

		int threads_per_block = 256; // Количество потоков в блоке (типичное значение)
		
		// Определяем количество блоков
		// Можно использовать эвристику, например, несколько блоков на каждый мультипроцессор (SM)
		// Либо просто покрыть n_steps, если оно не очень велико.
		// Ядро использует grid-stride loop, поэтому оно гибко к количеству блоков.
		int deviceId;
		cudaGetDevice(&deviceId);
		checkCudaError(cudaGetLastError(), "cudaGetDevice");

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceId);
		checkCudaError(cudaGetLastError(), "cudaGetDeviceProperties");

		int num_sms = prop.multiProcessorCount;
		std::cout << "Количество мультипроцессоров (SMs) на GPU: " << num_sms << std::endl;

		int num_blocks = prop.multiProcessorCount * 4; // Эвристика: 4 блока на SM
		// Если n_steps мало, скорректируем num_blocks, чтобы не было слишком много пустых блоков
		if (n_steps < num_blocks * threads_per_block) {
				num_blocks = (n_steps + threads_per_block - 1) / threads_per_block;
		}
		if (num_blocks == 0 && n_steps > 0) { // Гарантируем хотя бы один блок, если есть шаги
				num_blocks = 1;
		}
		
		std::cout << "Используется " << num_blocks << " блоков и " << threads_per_block << " потоков на блок." << std::endl;
		size_t shared_mem_size = threads_per_block * sizeof(double);
		std::cout << "Размер разделяемой памяти на блок: " << shared_mem_size << " байт." << std::endl;
		if (shared_mem_size > prop.sharedMemPerBlock) {
				std::cerr << "Ошибка: Требуемый размер разделяемой памяти (" << shared_mem_size 
									<< " байт) превышает доступный на блоке (" << prop.sharedMemPerBlock << " байт)." << std::endl;
				return 1;
		}


		// Выделение памяти на GPU для хранения сумм от каждого блока
		double* d_block_sums;
		cudaMalloc(&d_block_sums, num_blocks * sizeof(double));
		checkCudaError(cudaGetLastError(), "cudaMalloc d_block_sums");

		// Запуск CUDA ядра
		calculate_block_sums_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(a, b, n_steps, d_block_sums);
		checkCudaError(cudaGetLastError(), "kernel launch calculate_block_sums_kernel");
		
		// Синхронизация, чтобы убедиться, что ядро завершило выполнение перед копированием результатов
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError(), "cudaDeviceSynchronize after kernel");

		// Копирование результатов (сумм по блокам) с GPU на CPU
		std::vector<double> h_block_sums(num_blocks);
		cudaMemcpy(h_block_sums.data(), d_block_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaError(cudaGetLastError(), "cudaMemcpy d_block_sums to host");

		// Суммирование частичных сумм на CPU
		double total_sum_of_f_values = 0.0;
		for (int i = 0; i < num_blocks; ++i) {
				total_sum_of_f_values += h_block_sums[i];
		}

		// Финальный результат интеграла
		double integral_result = total_sum_of_f_values * dx;

		std::cout << std::fixed << std::setprecision(15); // Для вывода с высокой точностью
		std::cout << "Приближенное значение интеграла: " << integral_result << std::endl;

		if (std::isinf(integral_result) || std::isnan(integral_result)) {
				std::cout << "Результат INF или NaN указывает на то, что интеграл расходится, как и ожидалось из анализа функции." << std::endl;
		}
		
		cudaEventRecord(stop); // Запись конца
		cudaEventSynchronize(stop); // Ожидание завершения
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop); // Получение времени
		printf("\nЗатраченное время %f миллисекунд", milliseconds);

		// Освобождение памяти на GPU
		cudaFree(d_block_sums);
		checkCudaError(cudaGetLastError(), "cudaFree d_block_sums");

		return 0;
}

