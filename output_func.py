from math_func import mat_ozh, sko

import numpy as np
import plotly.express as px
from plotly.offline import plot
# МОЯ ЛИЧНАЯ МАЛЕНЬКАЯ БИБЛИОТЕКА В КОТОРОЙ ЛЕЖАТ ВСЕ РЕАЛИЗОВАННЫЕ ФУНКЦИИ

def errors_in_file(b, l, h, name=''):  # Функция вывода ошибок, мат.ож и ско по координатам в файл
    # аргумент фунции массивы ошибок по координатам и имя для имени файла
    m1 = mat_ozh(b)
    s1 = sko(b)
    o1 = abs(m1) + 2*s1

    m2 = mat_ozh(l)
    s2 = sko(l)
    o2 = abs(m2) + 2*s2

    m3 = mat_ozh(h)
    s3 = sko(h)
    o3 = abs(m3) + 2*s3
    if name == '':
        writing = open('errors.txt', 'w')
    else:
        writing = open(name, 'w')
    writing.write('Ошибка по ШИРОТЕ (В,м)' + '\n')
    writing.write('Математическое ожидание : ' + str(m1) + '\n')
    writing.write('СКО : ' + str(s1) + '\n')
    writing.write('Ошибка (95%) : ' + str(o1) + '\n')
    writing.write('Ошибка по ДОЛГОТЕ (L,м)' + '\n')
    writing.write('Математическое ожидание : ' + str(m2) + '\n')
    writing.write('СКО : ' + str(s2) + '\n')
    writing.write('Ошибка (95%) : ' + str(o2) + '\n')
    writing.write('Ошибка по ВЫСОТЕ (H,м)' + '\n')
    writing.write('Математическое ожидание : ' + str(m3) + '\n')
    writing.write('СКО : ' + str(s3) + '\n')
    writing.write('Ошибка (95%) : ' + str(o3) + '\n')

    writing.close()
    return 1


def errors_pr_in_file(psevdo_range):  # Функция вывода ошибок, мат.ож и ско по псевдодальностям в файл
    # аргумент фунции массив ошибок по псевдодальностям
    k = 0
    writing = open('pr_errors.txt', 'w')
    for i in psevdo_range:
        i = i.to_numpy()
        m = mat_ozh(i)
        s = sko(i)
        o = abs(m) + 2*s
        writing.write('Ошибка по псевдодальности (В,м) __' + str(k) + '\n')
        writing.write('Математическое ожидание : ' + str(m) + '\n')
        writing.write('СКО - ' + str(s) + '\n')
        writing.write('Ошибка (95%) - ' + str(o) + '\n\n')
        k += 1
    writing.close()
    return 1


def show_diff(x):  # вывод граффика дифференциала параметра
    fig_0 = px.line(x=range(len(x)-1), y=np.diff(x), title='Дифференциал')
    # fig_0.show()
    plot(fig_0)
    return 1


def fast_plot(x, name=''):  # Быстрый график на плотли экспресс
    fig_0 = px.line(x=range(len(x)), y=x, title='График '+name)
    fig_0.show()
    # plot(fig_0)
    return 1
