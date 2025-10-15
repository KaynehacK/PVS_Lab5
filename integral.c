#include <stdio.h>
#include <stdlib.h>
#include <math.h>     // Для M_PI, sin, cos, log, isinf, isnan
#include <time.h>     // Для измерения времени (если нужно, но в этом примере используем clock())

// Определяем M_PI, если он не определен в math.h (хотя обычно определен)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Функция для интегрирования f(x) ---
// f(x) = cot(x) / (ln(1+sin(x)) * sin(1+sin(x)))
// cot(x) = cos(x) / sin(x)
double cpu_f(double x) {
    double sin_x = sin(x);
    double cos_x = cos(x);

    // Знаменатель, член 1: log(1+sin(x))
    double term_1_plus_sin_x = 1.0 + sin_x;
    double log_term = log(term_1_plus_sin_x);

    // Знаменатель, член 2: sin(1+sin(x))
    double sin_term_denom = sin(term_1_plus_sin_x);

    double cot_x = cos_x / sin_x;

    // Проверка на ноль в знаменателе или sin(x) = 0
    if (sin_x == 0.0 || log_term == 0.0 || sin_term_denom == 0.0) {
        // Возвращаем +/- INF, если это сингулярность.
        // В C/C++ деление на 0.0 обычно возвращает INF или NaN.
        // Используем copysign для правильного знака INF, как в исходном коде
        if (sin_x == 0.0) return copysign(1.0 / 0.0, cos_x);
        // В других случаях просто позволяем произойти делению:
    }

    return cot_x / (log_term * sin_term_denom);
}

// --- Главная функция ---
int main() {
    double a = 0.0;
    double b = M_PI - 0.1;
    // Используем long long для n_steps, чтобы соответствовать размеру исходного кода
    long long n_steps = 100000000LL;

    // Используем clock() для измерения времени CPU
    clock_t start_time = clock();

    printf("Интегрирование f(x) = cot(x) / (ln(1+sin(x)) * sin(1+sin(x)))\n");
    printf("Метод: средних прямоугольников (последовательный)\n");
    printf("Интервал: [%.1f, %.1f]\n", a, b);
    printf("Количество шагов (N): %lld\n", n_steps);

    if (n_steps <= 0) {
        fprintf(stderr, "Количество шагов должно быть положительным.\n");
        return 1;
    }

    double dx = (b - a) / n_steps;
    double total_sum_of_f_values = 0.0;

    // Последовательный расчет интеграла методом средних прямоугольников
    for (long long i = 0; i < n_steps; ++i) {
        double x_mid = a + (i + 0.5) * dx; // Средняя точка i-го прямоугольника

        // Проверка границ интервала (как в исходном CUDA коде, хотя она не нужна для этого метода)
        if (x_mid > a && x_mid < b) {
            total_sum_of_f_values += cpu_f(x_mid);
        }
    }

    // Финальный результат интеграла
    double integral_result = total_sum_of_f_values * dx;

    printf("\n");
    // Вывод результата с высокой точностью (15 знаков после запятой)
    printf("Приближенное значение интеграла: %.15f\n", integral_result);

    if (isinf(integral_result) || isnan(integral_result)) {
        printf("Результат INF или NaN указывает на то, что интеграл расходится, как и ожидалось из анализа функции.\n");
    }

    // Расчет и вывод затраченного времени
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0; // Время в миллисекундах
    printf("\nЗатраченное время %f миллисекунд\n", time_spent);

    return 0;
}