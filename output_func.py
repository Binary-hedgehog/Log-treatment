'''
Модуль для вывода данных в файл или на экран
в виде графиков
'''
from math_func import mat_ozh, sko

import numpy as np
import plotly.express as px
from plotly.offline import plot

from numpy import ndarray
from typing import List, Dict, Tuple
from pandas.core.frame import DataFrame


def errors_in_file(b: ndarray, l: ndarray, 
                   h: ndarray, name: str=''):
    '''
    Функция вывода ошибок, мат.ож и ско по координатам в файл
    аргумент фунции массивы ошибок по координатам и имя для имени файла
    '''
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
    writing.write(f'Ошибка по ШИРОТЕ (В,м)\n'
                  f'Математическое ожидание : {m1}\n'
                  f'СКО : {s1}\n' 
                  f'Ошибка (95%) : {o1}\n' 
                  f'Ошибка по ДОЛГОТЕ (L,м)\n' 
                  f'Математическое ожидание : {m2}\n'
                  f'СКО : {s2}\n'
                  f'Ошибка (95%) : {o2}\n' 
                  f'Ошибка по ВЫСОТЕ (H,м)\n' 
                  f'Математическое ожидание : {m3}\n' 
                  f'СКО : {s3}\n'
                  f'Ошибка (95%) : {o3}\n')
    writing.close()
    return 1


def errors_pr_in_file(psevdo_range: List[List[float]]):
    '''
    Функция вывода ошибок, мат.ож и ско по псевдодальностям в файл
    аргумент фунции массив ошибок по псевдодальностям
    '''
    k = 0
    writing = open('pr_errors.txt', 'w')
    for i in psevdo_range:
        i = i.to_numpy()
        m = mat_ozh(i)
        s = sko(i)
        o = abs(m) + 2*s
        writing.write(f'Ошибка по псевдодальности (В,м) : {k}\n'
                      f'Математическое ожидание : {m}\n'
                      f'СКО : {s}\n'
                      f'Ошибка (95%) : {o}\n\n')
        k += 1
        
    writing.close()
    return 1


def show_diff(x: ndarray):
    # вывод граффика дифференциала параметра
    fig_0 = px.line(x=range(len(x)-1), y=np.diff(x), title='Дифференциал')
    # fig_0.show()
    plot(fig_0)
    return 1


def fast_plot(x: ndarray, name: str=''):
    # Быстрый график на плотли экспресс
    fig_0 = px.line(x=range(len(x)), y=x, title='График '+name)
    fig_0.show()
    # plot(fig_0)
    return 1
