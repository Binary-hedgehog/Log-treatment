#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import pi

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot

# import functions as fu
import math_func as mf
import output_func as of
import time_func as tf
import input_func as inf


print("Need to do")
print("Корректное обрезание по фалгу навигации ПНАП")
print("Добавить везде возможность строить графики (обс) не склеенные")
print("Протестировать работу всех функций на всех протоколах (кроме КБТИ)")
print("end")


class StartClass(object):
    '''
    Класс ввода и первичной подготовки данных
    Инициилизируется функцией input с выбором соответсвующего протокола
    для работы
    Имеет функции выводящие любые необходимые параметры
    '''

    def __init__(self):

        self.prot = 0  # Выбор протокола взаимодействия (rinex=1, pnap=2)
        # self.dict2 = fu.init_dict()# obs
        self.dict_kbti = inf.init_dict()    # kbti
        self.dict_type = inf.init_dict()    # sat type

        self.file_track = inf.init_array()  # track
        self.file_obs = inf.init_array()    # obs
        self.file_state1 = inf.init_array()  # state1
        self.file_state2 = inf.init_array()  # state2

        self.data_correction = False
        '''
        обрезает данные по адекватности флага навигации (1/0) - как будто уже и
        не сильно нужен...хотя надо бы его проработать
        '''
        self.in_nav_correction = False
        '''
         обрезает данные по dgop в пределах
        1<=gdop<=10 и по флагу навигации 1
        '''
        # self.kbti_nav = fu.init_array()    # kbti_nav
        # self.kbti_land = fu.init_array()   # kbti_land
        # self.kbti_etalon_flag = False

        df = pd.read_csv('IM-2/f_blh.txt')
        self.im_date = 0
        '''
        тип данных имитатора им-2, где 0-старый тип сценариев, с радианами и
        без загаловков, 1 - новый
        '''
        try:
            float(df.columns[0])
        except ValueError:
            self.im_date = 1

        if self.im_date:
            dec = '.'
        else:
            dec = ','

        self.f_blh = pd.read_csv('IM-2/f_blh.txt', sep='\s+', decimal=dec).to_numpy()
        self.f_xyz = pd.read_csv('IM-2/f_xyz.txt', sep='\s+', decimal=dec).to_numpy()
        self.f_frg = pd.read_csv('IM-2/frg.txt', sep='\s+', decimal=dec).to_numpy()
        self.f_frn = pd.read_csv('IM-2/frn.txt', sep='\s+', decimal=dec).to_numpy()
        # self.f_fvg = pd.read_csv('IM-2/f_fvg.txt', sep = '\s+', decimal = dec).to_numpy()
        # self.f_fvn = pd.read_csv('IM-2/f_fvn.txt', sep = '\s+', decimal = dec).to_numpy()

    def input(self, proto_type='rinex'):
        # , kbti=False, kbti_etalon_flag = 0, kbti_name = '', kbti_name_land = ''

        # kbti_etalon_flag - //без эталона - 0, с эталоном в одном файле - 1, с эталоном в двух файлах - 2\\

        if proto_type == 'rinex':
            self.prot = 1
            self.dict_type = dict(gps=1, glon=2, vt=4)
            self.file_track = pd.read_csv(
                'logs/rinex/track_0.txt', header=0, sep='\s+')
            inf.correct_obs_rinex()
            self.file_obs = pd.read_csv(
                'logs/rinex/obs.txt', header=0, sep='\s+')

            self.file_track, self.file_obs = tf.rinex_prepare_time(
                self.file_track, self.file_obs)
            self.file_track, self.file_obs = inf.rinex_prepare_data(
                self.file_track, self.file_obs)

        elif proto_type == 'pnap':
            self.prot = 2
            self.dict_type = dict(gps=0, bdu=1, glon=2, gal=3, glk=4,
                                  sbs=5)
            self.dict_time_type = dict(gps=0, bdu=1, glon=2, gal=3,
                                       utc=4, msk=5, rcv=6)
            self.file_track = pd.read_csv(
                'logs/pnap/track_0.txt', header=0, sep='\s+')
            self.file_obs = pd.read_csv(
                'logs/pnap/obs_0.txt', header=0, sep='\s+')
            self.file_state1 = pd.read_csv(
                'logs/pnap/state1.txt', header=0, sep='\s+')
            self.file_state2 = pd.read_csv(
                'logs/pnap/state2.txt', header=0, sep='\s+')

            # prepare time pnap!   !Need to test!
            self.file_track, self.file_state1 = tf.pnap_prepare_time(
                self.file_track, self.file_state1)
            self.file_track, self.file_state1, self.file_state2, self.file_obs = inf.pnap_prepare_data(
                self.file_track, self.file_state1, self.file_state2, self.file_obs)

        else:
            print('input - error name of protocol')
            return 0
        if self.prot == 1:
            self.num = 'num_track'
            self.ks_glo = self.take_parametr('qtyGlonInNav')
            self.ks_gps = self.take_parametr('qtyGpsInNav')
            self.diffmode = self.take_parametr('isOnDiffMode')
        else:
            self.num = 'num_track_bnm'
            self.ks_glo = self.take_parametr('qtyGlonL1InNav')
            self.ks_gps = self.take_parametr('qtyGpsL1InNav')
            self.ks_bdu = self.take_parametr('qtyBduB1InNav')
            self.ks_gal = self.take_parametr('qtyGalE1InNav')
            self.ks_sbas = self.take_parametr('qtySBASInNav')

    def sat_search(self, sat_type):
        '''
        Функция, которая выводит номера спутников и литеры для них
        (ГЛОНАСС СТ и ВТ)
        в виде списка, если они встречаются в сценарии

        Если список пустой, значит такой НС нет в сценарии
        '''
        sats = []
        letter = []

        try:
            frame1 = self.file_obs[self.file_obs['typeSat'] == self.dict_type[sat_type]]
        except KeyError:
            return sats, letter

        if sat_type == 'glon':

            for i in range(1, 64):

                if i in frame1['numSat'].to_numpy():
                    sats.append(i)
                    frame2 = frame1[frame1['numSat'] == i]

                    if sat_type == 'glon' or sat_type == 'vt':
                        let = set(frame2['letterSat'])
                        letter.append(let)
        else:
            sats = set(frame1['numSat'])

        return sats, letter

    def take_parametr(self, parametr, sat_number='all', sat_type='0',
                      data_type='low'):
        # Функция возвращает параметр из данных ППА или ПНАП

        if parametr == '0':
            return []

        if parametr == 'track':
            return self.file_track

        elif parametr == 'obs':
            return self.file_obs

        elif parametr == 'state1':
            return self.file_state1

        elif parametr == 'state2':
            return self.file_state2

        elif parametr in self.file_track.columns and sat_type == '0':  # track both

            if self.in_nav_correction:
                # обрезает данные по навигации 1<=gdop<=10, dflg==1
                frame0 = self.file_track
                frame1 = frame0[frame0['gdop'] <= 10]
                frame2 = frame1[frame1['gdop'] >= 1]
                frame3 = frame2[frame2['decisionFlag'] == 1]
                return frame3[parametr]

            return self.file_track[parametr]

        elif parametr in self.file_obs.columns and self.prot == 1:  # obs rinex

            if sat_number not in self.file_obs['numSat'] and sat_number != 'all':
                print("there is no sat with this number")
                return np.zeros(1)

            if self.dict_type[sat_type] not in self.file_obs['typeSat']:
                print("there is no sat with this type")
                return np.zeros(1)

            if sat_number == 'all':  # Значение параметра по всем спутникам

                sats = self.sat_search(sat_type)
                list1 = []

                for sat_i in sats[0]:
                    frame = self.file_obs[
                        (self.file_obs['numSat'] == sat_i) &
                        (self.file_obs['typeSat'] == self.dict_type[sat_type])]
                    list1.append(frame[parametr])
                return list1  # можно добавить вывод длины списка чиста по приколу#

            if data_type == 'full':
                first_input = 0

                for i in range(len(self.file_obs)):

                    if self.file_obs['numSat'][i] == sat_number and self.file_obs['typeSat'][i] == self.dict_type[sat_type]:
                        first_input = i
                        break

                while first_input > 46:
                    first_input -= 46
                slice_list = [i for i in range(first_input, int(len(self.file_obs) - 46), 46)]
                frame = self.file_obs.iloc[slice_list, :]

            else:
                frame = self.file_obs[
                    (self.file_obs['numSat'] == sat_number) &
                    (self.file_obs['typeSat'] == self.dict_type[sat_type])]

            return frame[parametr]

        elif parametr in self.file_obs.columns:  # PNAP obs

            if sat_number == 'all':
                arr1 = self.file_obs[self.file_obs['typeSat'] == self.dict_type[sat_type]]
                return arr1[parametr]

            else:
                arr1 = self.file_obs[(self.file_obs['typeSat'] == self.dict_type[sat_type])
                                     & (self.file_obs['numSat'] == sat_number)]
                return arr1[parametr]

        if self.prot == 2:

            if parametr in self.file_state1.columns:  # PNAP state1
                '''
                print('Данные из state1 были аппроксимированны')
                if self.in_nav_correction: # обрезает данные по 1<gdop<10
                    frame0 = self.file_state1
                    frame1 = frame0[frame0['gdop']<=10]
                    frame2 = frame1[frame1['gdop']>=1]
                else:
                '''
                frame2 = self.file_state1
                return mf.approx_linar(
                    frame2[parametr].to_numpy(),
                    self.file_track['time'].to_numpy(),
                    frame2['time'].to_numpy())

            elif parametr in self.file_state2.columns:  # PNAP state2
                print('Данные из state2 были аппроксимированны')
                return mf.approx_linar(
                    self.file_state2[parametr].to_numpy(),
                    self.file_track['time'].to_numpy(),
                    self.file_state2['currentTimeGlon'].to_numpy() - 10800)

        else:
            print("incorrect name of parametr")
            return 0


class LogClass(StartClass):
    '''
    Класс обработки логов
    Содержит различные фукции для построения графиков для различных задач
    Содержит типовые аргументы переменных
    sl_l - срез значений лога (графика) слева на n
    sl_r - срез значений лога (графика) справа на n
    data_corr - флаг коррекции данных по адекватности
    in_nav_corr - флаг коррекции данных по навигации
    xaxe - выбор типа оси Х для вывода графиков
    '''

    def distance(self, point_name, sl_l=0, sl_r=0, data_corr=False,
                 in_nav_corr=False, xaxe='num'):
        # Функция для вывода графика расстояния от некой точки
        '''
        Точка на вход подается списком длинной 3 или 6
        (см.пример в шапке модуля functions)
        '''
        if len(point_name) == 3:
            point_x = point_name[0]
            point_y = point_name[1]
            point_z = point_name[2]
        elif len(point_name) == 6:
            point_x = point_name[3]
            point_y = point_name[4]
            point_z = point_name[5]
        else:
            print('Error point lenght')
            return 0

        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        dflag = self.take_parametr('decisionFlag')

        i_start = 0
        i_end = len(gdop)

        i_start += sl_l
        i_end -= sl_r
        # print(i_start, i_end,'I')

        if xaxe == 'num':
            xaxe = self.take_parametr(self.num)
            if self.prot == 1:
                xaxe_name = "Номера отсчетов, 1/с"
            else:
                xaxe_name = "Номера отсчетов, 1/10с"
        elif xaxe == 'time':
            # xaxe = fu.sec_to_time(fu.time_remade(self.take_parametr('time').to_numpy()))
            xaxe = tf.sec_to_time(self.take_parametr('time').to_numpy())
            xaxe_name = "Время"
        elif xaxe == 'sec':
            # xaxe = fu.time_remade(self.take_parametr('time').to_numpy())
            xaxe = self.take_parametr('time').to_numpy()
            xaxe_name = "Время, с"
        else:
            print('incorrect input axe')
            return 0

        x = self.take_parametr('userX()')
        y = self.take_parametr('userY()')
        z = self.take_parametr('userZ()')

        r = np.sqrt((x - point_x)**2 + (y - point_y)**2 + (z - point_z)**2)  # дальность

        fig0 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.04, subplot_titles=(
                'Дальность, м',
                'Флаги навигации, дифф.режима и геометрический фактор'))

        fig0.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=r[i_start:i_end],
                name=''),
            row=1, col=1)
        fig0.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=dflag[i_start:i_end],
                name='Флаг навигации'),
            row=2, col=1)
        fig0.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=gdop[i_start:i_end],
                name='Геом.фактор'),
            row=2, col=1)
        if self.prot == 1:
            fig0.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.diffmode[i_start:i_end],
                    name='Дифф.режим'),
                row=2, col=1)
        fig0.update_layout(
            height=900, width=1900, title_text="График дальности")
        fig0.update_xaxes(title_text=xaxe_name, row=2, col=1)
        # fig0.show()
        plot(fig0)
        return 1

    def show_graph(self, parametr, num_sat='all', type_sat='0', sl_l=0, sl_r=0,
                   data_corr=False, in_nav_corr=False, xaxe='num'):
        # Функция для вывода графика от некоего параметра
        '''
        Список параметров можно взять из шапки подгружаемых файлов
        Для данных первички необходимо указывать параметры num_sat и type_sat
        Есть предустановленные параметры такие как: координаты, скорости,
        ускорения, рывки (включая данные имитатора) и тд.  (ниже по коду)
        '''
        if parametr == 'diff_time':
            of.fast_plot(np.diff(self.take_parametr('time').to_numpy()),
                         'дифференциала по времени')
            return 1

        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        dflag = self.take_parametr('decisionFlag')

        i_start = 0
        i_end = len(gdop)

        i_start += sl_l
        i_end -= sl_r
        print(i_start, i_end, 'I')

        if xaxe == 'num':
            xaxe = self.take_parametr(self.num)
            if self.prot == 1:
                xaxe_name = "Номера отсчетов, 1/с"
            else:
                xaxe_name = "Номера отсчетов, 1/10с"
        elif xaxe == 'time':
            # xaxe = fu.sec_to_time(fu.time_remade(self.take_parametr('time').to_numpy()))
            xaxe = tf.sec_to_time(self.take_parametr('time').to_numpy())
            xaxe_name = "Время"
        elif xaxe == 'sec':
            # xaxe = fu.time_remade(self.take_parametr('time').to_numpy())
            xaxe = self.take_parametr('time').to_numpy()
            xaxe_name = "Время, с"
        else:
            print('incorrect input xaxe')
            return 0

        if parametr in ['coor', 'rate', 'exeler', 'jerk',
                        'coor0', 'rate0', 'exeler0', 'jerk0']:
            if parametr in ['coor', 'rate', 'exeler', 'jerk']:
                x = self.take_parametr('userX()').to_numpy()
                y = self.take_parametr('userY()').to_numpy()
                z = self.take_parametr('userZ()').to_numpy()
                blh = mf.xyz_blh(x, y, z)

                if parametr == 'coor':
                    names = ["Координата х, м", "Координата у, м",
                             "Координата z, м",
                             "Геометрический фактор и флаг навигации",
                             "Количество спутников", "Координата b, град.",
                             "Координата l, град.", "Координата h, м"]
                    title1 = 'Координаты'

                elif parametr == 'rate':
                    names = ["Составляющая скорости vх, м/c",
                             "Составляющая скорости vу, м/c",
                             "Составляющая скорости vz, м/c",
                             "Геометрический фактор и флаг навигации",
                             "Количество спутников",
                             "Составляющая скорости vb, м/c",
                             "Составляющая скорости vl, м/c",
                             "Составляющая скорости vh, м/c"]
                    title1 = 'Скорости'

                    x = self.take_parametr('userVx').to_numpy()
                    y = self.take_parametr('userVy').to_numpy()
                    z = self.take_parametr('userVz').to_numpy()
                    blh = mf.lla2xyz_mas(x, y, z, blh[0], blh[1])

                elif parametr == 'exeler':
                    names = ["Ускорение vх, м/c2", "Ускорение vу, м/c2",
                             "Ускорение vz, м/c2",
                             "Геометрический фактор и флаг навигации",
                             "Количество спутников",
                             "Ускорение vb, м/c2", "Ускорение vl, м/c2",
                             "Ускорение vh, м/c2"]
                    title1 = 'Ускорения'

                    x = np.diff(self.take_parametr('userVx').to_numpy()) / 0.1
                    y = np.diff(self.take_parametr('userVy').to_numpy()) / 0.1
                    z = np.diff(self.take_parametr('userVz').to_numpy()) / 0.1
                    blh = mf.lla2xyz_mas(x, y, z, blh[0], blh[1])

                elif parametr == 'jerk':
                    names = ["Рывок vх, м/c3", "Рывок vу, м/c3",
                             "Рывок vz, м/c3",
                             "Геометрический фактор и флаг навигации",
                             "Количество спутников",
                             "Рывок vb, м/c3", "Рывок vl, м/c3",
                             "Рывок vh, м/c3"]
                    title1 = 'Рывки'

                    x = np.diff(np.diff(
                        self.take_parametr('userVx').to_numpy())) / 0.01
                    y = np.diff(np.diff(
                        self.take_parametr('userVy').to_numpy())) / 0.01
                    z = np.diff(np.diff(
                        self.take_parametr('userVz').to_numpy())) / 0.01
                    blh = mf.lla2xyz_mas(x, y, z, blh[0], blh[1])

            elif parametr in ['coor0', 'rate0', 'exeler0', 'jerk0']:
                dt = self.f_xyz[1, 0] - self.f_xyz[0, 0]
                xaxe = self.f_xyz[:, 0]
                i_end = len(self.f_xyz[:, 1]) - sl_r

                if parametr == 'coor0':
                    names = ["Координата х, м", "Координата у, м",
                             "Координата z, м", "Координата b, град.",
                             "Координата l, град.", "Координата h, м"]
                    title1 = 'Координаты имитатора'
                    x = self.f_xyz[:, 1]
                    y = self.f_xyz[:, 2]
                    z = self.f_xyz[:, 3]
                    if self.im_date:
                        blh = [self.f_blh[:, 1],
                               self.f_blh[:, 2],
                               self.f_blh[:, 3]]
                    else:
                        blh = [self.f_blh[:, 1] * 180 / pi,
                               self.f_blh[:, 2] * 180 / pi,
                               self.f_blh[:, 3]]

                elif parametr == 'rate0':
                    names = ["Составляющая скорости vх, м/c",
                             "Составляющая скорости vу, м/c",
                             "Составляющая скорости vz, м/c",
                             "Составляющая скорости vb, м/c",
                             "Составляющая скорости vl, м/c",
                             "Составляющая скорости vh, м/c"]
                    title1 = 'Скорости имитатора'
                    x = self.f_xyz[:, 4]
                    y = self.f_xyz[:, 5]
                    z = self.f_xyz[:, 6]
                    blh = [self.f_blh[:, 4], self.f_blh[:, 5], self.f_blh[:, 6]]

                elif parametr == 'exeler0':
                    names = ["Ускорение vх, м/c2", "Ускорение vу, м/c2",
                             "Ускорение vz, м/c2", "Ускорение vb, м/c2",
                             "Ускорение vl, м/c2", "Ускорение vh, м/c2"]
                    title1 = 'Ускорения имитатора'
                    x = np.diff(self.f_xyz[:, 4]) / dt
                    y = np.diff(self.f_xyz[:, 5]) / dt
                    z = np.diff(self.f_xyz[:, 6]) / dt
                    blh = [np.diff(self.f_blh[:, 4]) / dt,
                           np.diff(self.f_blh[:, 5]) / dt,
                           np.diff(self.f_blh[:, 6]) / dt]

                elif parametr == 'jerk0':
                    names = ["Рывок vх, м/c3", "Рывок vу, м/c3",
                             "Рывок vz, м/c3", "Рывок vb, м/c3",
                             "Рывок vl, м/c3", "Рывок vh, м/c3"]
                    title1 = 'Рывки имитатора'
                    x = np.diff(np.diff(self.f_xyz[:, 4])) / dt / dt
                    y = np.diff(np.diff(self.f_xyz[:, 5])) / dt / dt
                    z = np.diff(np.diff(self.f_xyz[:, 6])) / dt / dt
                    blh = [np.diff(np.diff(self.f_blh[:, 4])) / dt / dt,
                           np.diff(np.diff(self.f_blh[:, 5])) / dt / dt,
                           np.diff(np.diff(self.f_blh[:, 6])) / dt / dt]

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    subplot_titles=(names[0], names[1], names[2]))

                fig.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=x[i_start:i_end],
                        name=''),
                    row=1, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=y[i_start:i_end],
                        name=''),
                    row=2, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=z[i_start:i_end],
                        name=''),
                    row=3, col=1)
                fig.update_layout(
                    height=900, width=1900, title_text=title1 + " XYZ")
                fig.update_xaxes(title_text='time', row=3, col=1)
                # fig.show()
                plot(fig)

                fig1 = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    subplot_titles=(names[3], names[4], names[5]))

                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=blh[0][i_start:i_end],
                        name=''),
                    row=1, col=1)
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=blh[1][i_start:i_end],
                        name=''),
                    row=2, col=1)
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=blh[2][i_start:i_end],
                        name=''),
                    row=3, col=1)
                fig1.update_layout(
                    height=900, width=1900, title_text=title1 + " BLH")
                fig1.update_xaxes(title_text='time', row=3, col=1)
                # fig1.show()
                plot(fig1)
                return 1

            fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                subplot_titles=(
                    names[0], names[1], names[2], names[3], names[4]))

            fig.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=x[i_start:i_end],
                    name=''),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=y[i_start:i_end],
                    name=''),
                row=2, col=1)
            fig.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=z[i_start:i_end],
                    name=''),
                row=3, col=1)
            fig.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=gdop[i_start:i_end],
                    name="Геометрический фактор"),
                row=4, col=1)
            fig.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=dflag[i_start:i_end],
                    name="Флаг навигации"),
                row=4, col=1)
            fig.add_trace(
                go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_glo[i_start:i_end],
                        name="GLO"),
                row=5, col=1)
            fig.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_gps[i_start:i_end],
                    name="GPS"),
                row=5, col=1)
            if self.prot == 1:
                fig.add_trace(
                    go.Scatter(
                            x=xaxe[i_start:i_end],
                            y=self.diffmode[i_start:i_end],
                            name="Признак дифф.режима"),
                    row=4, col=1)
            if self.prot == 2:
                fig.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_gal[i_start:i_end],
                        name="GAL"),
                    row=5, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_bdu[i_start:i_end],
                        name="BDU"),
                    row=5, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_sbas[i_start:i_end],
                        name="SBAS"),
                    row=5, col=1)
            fig.update_layout(
                height=900, width=1900, title_text=title1 + " XYZ")
            fig.update_xaxes(title_text=xaxe_name, row=5, col=1)
            # fig.show()
            plot(fig)

            fig1 = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                subplot_titles=(
                    names[5], names[6], names[7], names[3], names[4]))

            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=blh[0][i_start:i_end],
                    name=''),
                row=1, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=blh[1][i_start:i_end],
                    name=''),
                row=2, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=blh[2][i_start:i_end],
                    name=''),
                row=3, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=gdop[i_start:i_end],
                    name="Геометрический фактор"),
                row=4, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=dflag[i_start:i_end],
                    name="Флаг навигации"),
                row=4, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_glo[i_start:i_end],
                    name="GLO"),
                row=5, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_gps[i_start:i_end],
                    name="GPS"),
                row=5, col=1)
            if self.prot == 1:
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.diffmode[i_start:i_end],
                        name="Признак дифф.режима"),
                    row=4, col=1)
            if self.prot == 2:
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_gal[i_start:i_end],
                        name="GAL"),
                    row=5, col=1)
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_bdu[i_start:i_end],
                        name="BDU"),
                    row=5, col=1)

            fig1.update_layout(
                height=900, width=1900, title_text=title1 + " BLH")
            fig1.update_xaxes(title_text=xaxe_name, row=5, col=1)
            # fig1.show()
            plot(fig1)
            return 1

        elif parametr in self.file_track.columns:  # ok
            fig = px.line(
                self.file_track[i_start:i_end], x=self.num, y=parametr,
                title='График от ' + str(parametr))
            # fig.show()
            plot(fig)
            return 1

        elif parametr in self.file_obs.columns:  # ok
            if num_sat != 'all' and type(num_sat) != int:
                print('incorrect num_sat')
                return 0
            if type_sat not in ['glon', 'gps', 'bdu', 'gal']:
                print('incorrect type_sat')
                return 0
            if num_sat == 'all':
                sats = self.sat_search(type_sat)

                for i in sats[0]:

                    title1 = 'График параметра ' + str(parametr) + ' Спутник ' \
                        + type_sat + ' №' + str(i)
                    if type_sat == 'glon':
                        title1 += ' литера №' + str(sats[1][sats[0].index(i)])
                    df = self.file_obs
                    df = df[(df['typeSat'] == self.dict_type[type_sat]) & (df['numSat'] == i)]
                    fig = px.line(
                        df[i_start:i_end], x=df[i_start:i_end].index,
                        y=parametr, labels={'x': 'time', 'y': parametr},
                        title=title1)
                    # fig.show()
                    plot(fig)
                else:
                    print('incorrect input')
                    return 0
                return 1
            else:
                title1 = 'График параметра ' + str(parametr) + ' Спутник ' \
                    + type_sat + ' №' + str(num_sat)
                df = self.file_obs
                df = df[(df['typeSat'] == self.dict_type[type_sat]) & (df['numSat'] == num_sat)]
                fig = px.line(df[i_start:i_end], x=df[i_start:i_end].index,
                              y=parametr, title=title1)
                # fig.show()
                plot(fig)
                return 1

        elif parametr in self.file_state1.columns:  # ok
            fig = px.line(
                self.file_state1[i_start:i_end], x='num_state1_bnm',
                y=parametr, title='График от ' + str(parametr))
            # fig.show()
            plot(fig)
            return 1

        elif parametr in self.file_state2.columns:  # ok
            fig = px.line(
                self.file_state2[i_start:i_end], x='num_state2_bnm',
                y=parametr, title='График от ' + str(parametr))
            # fig.show()
            plot(fig)
            return 1

        else:
            print("incorrect parametr name")
            return 0

    def show_3d_plot(self, sl_l=0, sl_r=0, data_corr=False, in_nav_corr=False,
                     point=True):
        # Функция для вывода графика координат в трехмерной проекции
        '''
        '''
        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        i_start = 0
        i_end = len(gdop)

        i_start += sl_l
        i_end -= sl_r
        # print(i_start, i_end,'I')

        x = self.take_parametr('userX()').to_numpy()[i_start:i_end]
        y = self.take_parametr('userY()').to_numpy()[i_start:i_end]
        z = self.take_parametr('userZ()').to_numpy()[i_start:i_end]

        if point:
            x -= np.mean(x)
            y -= np.mean(y)
            z -= np.mean(z)

            fig = px.scatter_3d(x=x, y=y, z=z, title='XYZ in 3D')
            # fig.show()
            plot(fig)
        else:
            blh = mf.xyz_blh(x, y, z)
            fig = px.line_3d(x=x, y=y, z=z, title='XYZ in 3D')
            # fig.show()
            plot(fig)

            fig1 = px.line_3d(x=blh[0], y=blh[1], z=blh[2],
                              title='BLH in 3D')
            # fig1.show()
            plot(fig1)

        return 1

    def difference_by(self, point_val, parametr='coor', get_date=True, sl_l=0,
                      sl_r=0, by_point=False, in_nav_corr=False,
                      data_corr=False, xaxe='num', modificated=1,
                      show_graphs=True, diff_time=False):
        # Функция для вывода графиков ошибок по неким параметрам
        '''
        Параметром могут являться координаты или скорости
        get_date - флаг возвращения данных в виде списка
        by_point - построение ошибок от конкретной точки
        point_val - значение конкретной точки (список)
        modificated - множитель для модификации выходных данных
        show_graphs - флаг построения графиков
        diff_time - флаг вывода графика дельты времени, вместо gdop
        '''
        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        dflag = self.take_parametr('decisionFlag')

        gdop = self.take_parametr('gdop')

        i_start = 0
        i_end = len(gdop)

        i_start += sl_l
        i_end -= sl_r
        # print(i_start, i_end,'I')

        if xaxe == 'num':
            xaxe = self.take_parametr(self.num)
            if self.prot == 1:
                xaxe_name = "Номера отсчетов, 1/с"
            else:
                xaxe_name = "Номера отсчетов, 1/10с"
        elif xaxe == 'time':
            # xaxe = fu.sec_to_time(fu.time_remade(self.take_parametr('time').to_numpy()))
            xaxe = tf.sec_to_time(self.take_parametr('time').to_numpy())
            xaxe_name = "Время"
        elif xaxe == 'sec':
            # xaxe = fu.time_remade(self.take_parametr('time').to_numpy())
            xaxe = self.take_parametr('time').to_numpy()
            xaxe_name = "Время, с"
        else:
            print('incorrect input axe')
            return 0

        if not by_point:
            time = tf.time_chek(self.f_xyz[:, 0],
                                (self.take_parametr('time').to_numpy()))
        else:
            time = self.take_parametr('time').to_numpy()

        x = self.take_parametr('userX()').to_numpy()
        y = self.take_parametr('userY()').to_numpy()
        z = self.take_parametr('userZ()').to_numpy()
        b, l, h = mf.xyz_blh(x, y, z)

        if by_point:  # сравнение с точкой

            if point_val == 'mat_ozh':
                # point_val = [0 for point_val in range(len(point_val))]
                point_val[0] = mf.mat_ozh(b)
                point_val[1] = mf.mat_ozh(l)
                point_val[2] = mf.mat_ozh(h)
                if len(point_val) > 3:
                    point_val[3] = mf.mat_ozh(x)
                    point_val[4] = mf.mat_ozh(y)
                    point_val[5] = mf.mat_ozh(z)

            b_dif = (b - point_val[0]) * 3600 * 31  # переврод в метры
            l_dif = (l - point_val[1]) * 3600 * 19  # переврод в метры
            h_dif = h - point_val[2]

        else:
            b0, l0, h0 = mf.approx3(self.f_blh[:, 1], self.f_blh[:, 2],
                                    self.f_blh[:, 3], time, self.f_blh[:, 0])

            # fu.fast_plot(blh[:, 1], 'b0_im-2')
            # fu.fast_plot(b0, 'b0')
            if b0[0] >= 2 * pi:
                b_dif = b0 - b
                l_dif = l0 - l
            else:
                b_dif = b0*180/pi - b
                l_dif = l0*180/pi - l
            h_dif = h0 - h
            b_dif = b_dif * 3600 * 31  # переврод в метры
            l_dif = l_dif * 3600 * 19  # переврод в метры

        if parametr == 'coor':
            names = ["Ошибка координаты х, м", "Ошибка координаты у, м",
                     "Ошибка координаты z, м",
                     "Геометрический фактор и флаг навигации",
                     "Количество спутников", "Ошибка координаты b, м",
                     "Ошибка координаты l, м", "Ошибка координаты h, м"]
            output = []
            title1 = 'Ошибки по координатам'

            if len(point_val) > 3:
                if by_point:
                    x_dif = x - point_val[3]
                    y_dif = y - point_val[4]
                    z_dif = z - point_val[5]
                else:
                    x0, y0, z0 = mf.approx3(
                        self.f_xyz[:, 1], self.f_xyz[:, 2], self.f_xyz[:, 3],
                        time, self.f_xyz[:, 0])
                    x_dif = x0 - x
                    y_dif = y0 - y
                    z_dif = z0 - z

            output.append(b_dif * modificated)
            output.append(l_dif * modificated)
            output.append(h_dif * modificated)
            if len(point_val) > 3:
                output.append(x_dif * modificated)
                output.append(y_dif * modificated)
                output.append(z_dif * modificated)

        elif parametr == 'rate':
            names = ["Ошибка составляющей скорости vх, м/с",
                     "Ошибка составляющей скорости vу, м/с",
                     "Ошибка составляющей скорости vz, м/с",
                     "Геометрический фактор и флаг навигации",
                     "Количество спутников",
                     "Ошибка составляющей скорости vb, м/с",
                     "Ошибка составляющей скорости vl, м/с",
                     "Ошибка составляющей скорости vh, м/с"]

            output = []
            title1 = 'Ошибки по скоростям'

            if by_point:
                vx_dif = self.take_parametr('userVx').to_numpy()
                vy_dif = self.take_parametr('userVy').to_numpy()
                vz_dif = self.take_parametr('userVz').to_numpy()
            else:
                vx0, vy0, vz0 = mf.approx3(
                    self.f_xyz[:, 4], self.f_xyz[:, 5], self.f_xyz[:, 6],
                    time, self.f_xyz[:, 0])
                vx_dif = vx0 - self.take_parametr('userVx').to_numpy()
                vy_dif = vy0 - self.take_parametr('userVy').to_numpy()
                vz_dif = vz0 - self.take_parametr('userVz').to_numpy()

            vblh_dif = mf.lla2xyz_mas(vx_dif, vy_dif, vz_dif, b_dif, l_dif)

            output.append(vblh_dif[0] * modificated)
            output.append(vblh_dif[1] * modificated)
            output.append(vblh_dif[2] * modificated)
            output.append(vx_dif * modificated)
            output.append(vy_dif * modificated)
            output.append(vz_dif * modificated)

        if get_date and not show_graphs:
            of.errors_in_file(
                output[0][i_start:i_end], output[1][i_start:i_end],
                output[2][i_start:i_end])
            return output
        
        if diff_time:
            names[3] = 'Дифференциал времени'
            names[4] = "Геометрический фактор, флаг навигации, Количество спутников"
        fig1 = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=(names[5], names[6], names[7], names[3], names[4]))

        fig1.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=output[0][i_start:i_end],
                name=''),
            row=1, col=1)
        fig1.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=output[1][i_start:i_end],
                name=''),
            row=2, col=1)
        fig1.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=output[2][i_start:i_end],
                name=''),
            row=3, col=1)
        if not diff_time:
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=gdop[i_start:i_end],
                    name="Геом. фактор"),
                row=4, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=dflag[i_start:i_end],
                    name="Флаг навигации"),
                row=4, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_glo[i_start:i_end],
                    name="GLO"),
                row=5, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_gps[i_start:i_end],
                    name="GPS"),
                row=5, col=1)
            if self.prot == 1:
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.diffmode[i_start:i_end],
                        name="Признак дифф.режима"),
                    row=4, col=1)
            if self.prot == 2:
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_gal[i_start:i_end],
                        name="GAL"),
                    row=5, col=1)
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_bdu[i_start:i_end],
                        name="BDU"),
                    row=5, col=1)
                fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                          y=self.ks_sbas[i_start:i_end],
                                          name="SBAS"),
                               row=5, col=1)
        else:
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=np.diff(time)[i_start:i_end],
                    name=""),
                row=4, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=gdop[i_start:i_end],
                    name="Геом. фактор"),
                row=5, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=dflag[i_start:i_end],
                    name="Флаг навигации"),
                row=5, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_glo[i_start:i_end],
                    name="GLO"),
                row=5, col=1)
            fig1.add_trace(
                go.Scatter(
                    x=xaxe[i_start:i_end],
                    y=self.ks_gps[i_start:i_end],
                    name="GPS"),
                row=5, col=1)
            if self.prot == 2:
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_glo[i_start:i_end],
                        name="GAL"),
                    row=5, col=1)
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_gps[i_start:i_end],
                        name="BDU"),
                    row=5, col=1)
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.ks_sbas[i_start:i_end],
                        name="SBAS"),
                    row=5, col=1)
            else:
                fig1.add_trace(
                    go.Scatter(
                        x=xaxe[i_start:i_end],
                        y=self.diffmode[i_start:i_end],
                        name="Признак дифф.режима"),
                    row=5, col=1)

            fig1.update_layout(
                height=900, width=1900, title_text=title1 + " BLH")
        fig1.update_xaxes(title_text=xaxe_name, row=5, col=1)
        # fig1.show()
        plot(fig1)

        if get_date:
            of.errors_in_file(
                output[0][i_start:i_end], output[1][i_start:i_end],
                output[2][i_start:i_end])
            return output

        return 1

    def accordance(self, parametr='coor', sys='blh', sl_l=0, sl_r=0,
                   data_corr=False, in_nav_corr=False, xaxe='num'):
        # Функция для вывода графика сопоставлений по координатам/скоростям с графиком ошибок
        '''
        '''
        if sys not in ['blh', 'xyz']:
            print('incorrect sys type')
            return 0
        if parametr not in ['coor', 'rate']:
            print('incorrect parametr')
            return 0

        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        dflag = self.take_parametr('decisionFlag')
        i_start = 0
        i_end = len(gdop)

        i_start += sl_l
        i_end -= sl_r
        # print(i_start, i_end,'I')

        if xaxe == 'num':
            xaxe = self.take_parametr(self.num)
            if self.prot == 1:
                xaxe_name = "Номера отсчетов, 1/с"
            else:
                xaxe_name = "Номера отсчетов, 1/10с"
        elif xaxe == 'time':
            # xaxe = fu.sec_to_time(fu.time_remade(self.take_parametr('time').to_numpy()))
            xaxe = tf.sec_to_time(self.take_parametr('time').to_numpy())
            xaxe_name = "Время"
        elif xaxe == 'sec':
            # xaxe = fu.time_remade(self.take_parametr('time').to_numpy())
            xaxe = self.take_parametr('time').to_numpy()
            xaxe_name = "Время, с"
        else:
            print('incorrect input xaxe')
            return 0

        dif_full = self.difference_by(
            parametr, show_graphs=False, sl_l=0, sl_r=0)

        if parametr == 'coor':
            names = ["Координата ", "Истинная координата ",
                     "Ошибка по координате ",
                     "Геометрический фактор и флаг навигации",
                     "Количество спутников", ', м']
            title1 = ' координаты '

            if sys == 'xyz':
                parm_1 = self.take_parametr('userX()')
                parm_2 = self.take_parametr('userY()')
                parm_3 = self.take_parametr('userZ()')

                dif_1 = dif_full[0]
                dif_2 = dif_full[1]
                dif_3 = dif_full[2]
            else:
                blh = mf.xyz_blh(self.take_parametr('userX()'),
                                 self.take_parametr('userY()'),
                                 self.take_parametr('userZ()'))

                parm_1 = blh[0] * 3600 * 31  # переврод в метры
                parm_2 = blh[1] * 3600 * 19  # переврод в метры
                parm_3 = blh[2]

                dif_1 = dif_full[3]
                dif_2 = dif_full[4]
                dif_3 = dif_full[5]

        if parametr == 'rate':
            names = ["Скорость ", "Истинная скорость ", "Ошибка по скорости ",
                     "Геометрический фактор и флаг навигации",
                     "Количество спутников", ', м/с']
            title1 = ' составляющей скорости '

            if sys == 'xyz':
                parm_1 = self.take_parametr('userVx')
                parm_2 = self.take_parametr('userVy')
                parm_3 = self.take_parametr('userVz')

                dif_1 = dif_full[0]
                dif_2 = dif_full[1]
                dif_3 = dif_full[2]
            else:
                vblh = mf.lla2xyz_mas(
                    self.take_parametr('userVx'),
                    self.take_parametr('userVy'),
                    self.take_parametr('userVz'),
                    mf.xyz_blh(
                        self.take_parametr('userX()'),
                        self.take_parametr('userY()'),
                        self.take_parametr('userZ()'))[0],
                    mf.xyz_blh(
                        self.take_parametr('userX()'),
                        self.take_parametr('userY()'),
                        self.take_parametr('userZ()'))[1])

                parm_1 = vblh[0]
                parm_2 = vblh[1]
                parm_3 = vblh[2]

                dif_1 = dif_full[3]
                dif_2 = dif_full[4]
                dif_3 = dif_full[5]

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=(
                names[0]+sys[0]+names[5],
                names[2] + sys[0] + names[5],
                names[3], names[4]))

        fig.add_trace(
            go.Scatter(
                x=xaxe[i_start:i_end],
                y=parm_1[i_start:i_end],
                name=''),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=dif_1[i_start:i_end],
                                 name=''),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=gdop[i_start:i_end],
                                 name="Геометрический фактор"),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=dflag[i_start:i_end],
                                 name="Флаг навигации"),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=self.ks_glo[i_start:i_end],
                                 name="GLO"),
                      row=4, col=1)
        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=self.ks_gps[i_start:i_end],
                                 name="GPS"),
                      row=4, col=1)
        if self.prot == 2:
            fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                     y=self.ks_gal[i_start:i_end],
                                     name="GAL"),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                     y=self.ks_bdu[i_start:i_end],
                                     name="BDU"),
                          row=4, col=1)
        fig.update_layout(height=900, width=1900,
                          title_text="Соответствие" + title1 + sys[0])
        fig.update_xaxes(title_text=xaxe_name, col=1, row=4)
        fig.show()
        # plot(fig)

        fig1 = make_subplots(rows=4, cols=1,
                             shared_xaxes=True,
                             vertical_spacing=0.04,
                             subplot_titles=(names[0]+sys[1]+names[5],
                                             names[2]+sys[1]+names[5],
                                             names[3], names[4]))

        fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=parm_2[i_start:i_end],
                                  name=''),
                       row=1, col=1)
        fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=dif_2[i_start:i_end],
                                  name=''),
                       row=2, col=1)
        fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=gdop[i_start:i_end],
                                  name="Геометрический фактор"),
                       row=3, col=1)
        fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=dflag[i_start:i_end],
                                  name="Флаг навигации"),
                       row=3, col=1)
        fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=self.ks_glo[i_start:i_end],
                                  name="GLO"),
                       row=4, col=1)
        fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=self.ks_gps[i_start:i_end],
                                  name="GPS"),
                       row=4, col=1)
        if self.prot == 2:
            fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                      y=self.ks_gal[i_start:i_end],
                                      name="GAL"),
                           row=4, col=1)
            fig1.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                      y=self.ks_bdu[i_start:i_end],
                                      name="BDU"),
                           row=4, col=1)
        fig1.update_layout(height=900, width=1900,
                           title_text="Соответствие" + title1 + sys[1])
        fig1.update_xaxes(title_text=xaxe_name, col=1, row=4)
        fig1.show()
        # plot(fig1)

        fig2 = make_subplots(rows=4, cols=1,
                             shared_xaxes=True,
                             vertical_spacing=0.04,
                             subplot_titles=(names[0]+sys[2]+names[5],
                                             names[2]+sys[2]+names[5],
                                             names[3], names[4]))

        fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=parm_3[i_start:i_end],
                                  name=''),
                       row=1, col=1)
        fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=dif_3[i_start:i_end],
                                  name=''),
                       row=2, col=1)
        fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=gdop[i_start:i_end],
                                  name="Геометрический фактор"),
                       row=3, col=1)
        fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=dflag[i_start:i_end],
                                  name="Флаг навигации"),
                       row=3, col=1)
        fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=self.ks_glo[i_start:i_end],
                                  name="GLO"),
                       row=4, col=1)
        fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                  y=self.ks_gps[i_start:i_end],
                                  name="GPS"),
                       row=4, col=1)
        if self.prot == 2:
            fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                      y=self.ks_gal[i_start:i_end],
                                      name="GAL"),
                           row=4, col=1)
            fig2.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                      y=self.ks_bdu[i_start:i_end],
                                      name="BDU"),
                           row=4, col=1)
        fig2.update_layout(height=900, width=1900,
                           title_text="Соответствие" + title1 + sys[2])
        fig2.update_xaxes(title_text=xaxe_name, col=1, row=4)
        fig2.show()
        # plot(fig2)
        return 1

    def pvesdo_range(self, graph_show=True, get_date=False, type_sat='glon',
                     sl_l=0, sl_r=0, data_corr=False, in_nav_corr=False):
        # Функция для вывода ошибок по псевдодальностям
        '''
        graph_show - флаг построения графика
        get_date - флаг возвращения данных ошибок
        type_sat - тип СНС
        '''
        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        i_start = 0
        i_end = len(gdop)
        print('!!!Проверить корректность срезов в псевдодальностях')
        i_start += sl_l
        i_end -= sl_r
        print(i_start, i_end, 'I')

        sats = self.sat_search(sat_type=type_sat)[0]
        if sats == []:
            print('no sats ' + type_sat)
            return 0

        if type_sat == 'glon':
            im_date = self.f_frg
            koef = mf.koef_glon()
            try:
                T0 = self.take_parametr('T0Glon').to_numpy()
            except AttributeError:
                T0 = self.take_parametr('T0Glon')
        elif type_sat == 'gps':
            im_date = self.f_frn
            koef = mf.koef_gps()
            try:
                T0 = self.take_parametr('T0Gps').to_numpy()
            except AttributeError:
                T0 = self.take_parametr('T0Gps')
        else:
            print('incorrect type of sat')
            return 0
        # print(type(T0))
        # if type(T0) != 'numpy.ndarray':
        #    T0=T0.to_numpy()
        psevdo_range = []
        n = 0
        #####
        print('Какой-то косяк со временем!!! Псевдодальности!!!')
        time_name = 'time'  # time_name = 'time' - Rinex
        #####
        for i in sats:

            n += 1  # Для графиков
            if n > 4:
                n = 1
            try:
                time = tf.time_chek(im_date[:, 0],
                                    self.take_parametr(time_name, i, type_sat).to_numpy())
                sat_range = mf.psevdorange(im_date[:, i], time, im_date[:, 0],
                                           tf.time_chek(im_date[:, 0],
                                                        self.take_parametr('time')).to_numpy(), T0)
            except AttributeError:
                time = tf.time_chek(im_date[:, 0], self.take_parametr(time_name, i, type_sat))
                sat_range = mf.psevdorange(im_date[:, i], time, im_date[:, 0],
                                           tf.time_chek(im_date[:, 0],
                                           self.take_parametr('time')), T0)

            pr = self.take_parametr("prL1Sat", i, type_sat) * koef - sat_range
            psevdo_range.append(pr)
            if n == 1:
                fig1 = make_subplots(rows=4, cols=1)

            fig1.add_trace(go.Scatter(x=[j for j in range(len(pr[i_start:i_end]))],
                                      y=pr[i_start:i_end],
                                      name='Спутник №' + str(i)),
                           row=n, col=1)
            if n == 4:
                fig1.update_layout(height=900, width=1900,
                                   title_text="Ошибка псевдодальностей спутников " + type_sat + ", м")
                fig1.update_xaxes(title_text="Номера отсчетов, 1/с")
                if graph_show:
                    # fig1.show()
                    plot(fig1)

        if get_date:  # вывод df с ошибкой по псевдодальностям
            of.errors_pr_in_file(psevdo_range)
            return psevdo_range

    def plot_from_few_parms(self, name1='0', name2='0', name3='0', name4='0',
                            name5='0', sl_l=0, sl_r=0, data_corr=False,
                            in_nav_corr=False):
        # Функция для вывода графика от нескольких желаемых параметров
        '''
        Функция не сработает с данными из файло obs(первичка) т.к. там необходимо задавать
        тип и номер СНС
        Это пока не предусмотрено
        '''
        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        if self.prot == 1:
            num = 'num_track'
        else:
            num = 'num_track_bnm'
        num = self.take_parametr(num)
        i_start = 0
        i_end = len(num)

        i_start += sl_l
        i_end -= sl_r
        print(i_start, i_end, 'I')

        num_graphs = 0
        names = [name1, name2, name3, name4, name5]
        for i in names:
            if i != '0':
                num_graphs += 1

        if num_graphs == 0:
            print('0 parametrs choose')
            return 0

        parm1 = self.take_parametr(name1)  # .to_numpy()
        parm2 = self.take_parametr(name2)
        parm3 = self.take_parametr(name3)
        parm4 = self.take_parametr(name4)
        parm5 = self.take_parametr(name5)

        fig = make_subplots(rows=num_graphs, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.04)

        fig.add_trace(go.Scatter(x=num[i_start:i_end],
                                 y=parm1[i_start:i_end],
                                 name=name1),
                      row=1, col=1)
        if num_graphs > 1:
            fig.add_trace(go.Scatter(x=num[i_start:i_end],
                                     y=parm2[i_start:i_end],
                                     name=name2),
                          row=2, col=1)
        if num_graphs > 2:
            fig.add_trace(go.Scatter(x=num[i_start:i_end],
                                     y=parm3[i_start:i_end],
                                     name=name3),
                          row=3, col=1)
        if num_graphs > 3:
            fig.add_trace(go.Scatter(x=num[i_start:i_end],
                                     y=parm4[i_start:i_end],
                                     name=name4),
                          row=4, col=1)
        if num_graphs == 5:
            fig.add_trace(go.Scatter(x=num[i_start:i_end],
                                     y=parm5[i_start:i_end],
                                     name=name5),
                          row=5, col=1)
        fig.update_layout(height=900, width=1900)
        fig.update_xaxes(title_text="Номера отсчетов, 1/с", row=num_graphs, col=1)
        # fig.show()
        plot(fig)

    def coor_on_map(self, sl_l=0, sl_r=0, data_corr=False, in_nav_corr=False,
                    with_diff=False):
        # Функция для вывода графика координат на карте мира
        '''
        Формат Карты OSM
        with_diff - позволяет добавить маркировку наличия дифференциального режима
        во время полета
        '''
        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        # dflag = self.take_parametr('decisionFlag')

        x = self.take_parametr('userX()')
        y = self.take_parametr('userY()')
        z = self.take_parametr('userZ()')
        blh = mf.xyz_blh(x, y, z)

        i_start = 0
        i_end = len(gdop)

        if self.prot == 1:
            diffmode = self.take_parametr('isOnDiffMode').to_numpy()

        if with_diff:
            sss = []
            for i in range(len(diffmode)):
                if diffmode[i] == 1:
                    sss.append(i)
            B = blh[0].to_numpy()
            B = B[sss]
            L = blh[1].to_numpy()
            L = L[sss]

        i_start += sl_l
        i_end -= sl_r
        #print(i_start, i_end,'I')

        fig = go.Figure(go.Scattermapbox(mode="markers+lines",
                                         lat=blh[0][i_start:i_end],
                                         lon=blh[1][i_start:i_end],
                                         marker=go.scattermapbox.Marker(size=8)))
        if with_diff:
            fig.add_trace(go.Scattermapbox(mode="markers+lines",
                                           lat=B[i_start:i_end],
                                           lon=L[i_start:i_end],
                                           marker={'size': 10}))

        fig.update_layout(margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                          mapbox={'center': {'lon': 0, 'lat': 0},
                                  'style': "stamen-terrain",
                                  'center': {'lon': 0, 'lat': 0},
                                  'zoom': 1})
        # fig.show()
        plot(fig)

    def coor_on_map_px(self, sl_l=0, sl_r=0, data_corr=False, in_nav_corr=False):
        # Функция для вывода графика координат на карте мира
        '''
        Формат Карты OSM
        Функция идентична предыдущей, однако реализована на другой функции
        '''

        frame0 = self.take_parametr('track')

        if data_corr:
            frame1 = frame0[frame0['decisionFlag'] <= 1]
            frame0 = frame1[frame1['decisionFlag'] >= 0]

        if in_nav_corr:
            frame1 = frame0[frame0['gdop'] <= 10]
            frame0 = frame1[frame1['gdop'] >= 1]

        frame0['userX()'], frame0['userY()'], frame0['userZ()'] = mf.xyz_blh(frame0['userX()'],
                                                                             frame0['userY()'],
                                                                             frame0['userZ()'])
        frame0.rename(columns={'userX()': 'lat'}, inplace=True)
        frame0.rename(columns={'userY()': 'lon'}, inplace=True)
        frame0.rename(columns={'userZ()': 'hight'}, inplace=True)

        fig = px.scatter_mapbox(frame0, lat="lat", lon="lon",
                                hover_data=["hight", 'num_track'])
        fig.update_traces(mode='markers+lines')
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=4,
                          mapbox_center_lat=41,
                          margin={"r": 0, "t": 0, "l": 0, "b": 0})
        # fig.show()
        plot(fig)

    def snr_plot(self):
        # функция для красивого вывода СШ на график
        '''
        '''
        print('to_do')
        print('Протестировать =)')
        sats_glon = self.sat_search('glon')[0]
        sats_gps = self.sat_search('gps')[0]
        if self.prot == 2:
            sats_bdu = self.sat_search('bdu')[0]
            sats_gal = self.sat_search('gal')[0]

        if self.prot == 1:
            if len(sats_glon) != 0 and len(sats_gps) != 0:
                fig = make_subplots(rows=2, cols=1,
                                    vertical_spacing=0.1,
                                    shared_xaxes=True,
                                    subplot_titles=('Спутники ГЛОНАСС',
                                                    'Спутники GPS'))
                for i in sats_glon:
                    fig.add_trace(go.Scatter(x=[j for j in range(int(len((self.file_obs) / 46)))],
                                             y=self.take_parametr('snrSat', sat_number=i,
                                                                  sat_type='glon',
                                                                  data_type='full'),
                                             name='№' + str(i)),
                                  row=1, col=1)
                for i in sats_gps:
                    fig.add_trace(go.Scatter(x=[j for j in range(int(len((self.file_obs) / 46)))],
                                             y=self.take_parametr('snrSat', sat_number=i,
                                                                  sat_type='gps',
                                                                  data_type='full'),
                                             name='№' + str(i)),
                                  row=2, col=1)
                fig.update_layout(height=900, width=1900,
                                  title_text="СШ спутников")
                fig.update_xaxes(title_text="Время, с")
                # fig.show()
                plot(fig)

            elif len(sats_glon) != 0:
                fig = make_subplots(rows=1, cols=1, 
                                    subplot_titles=('Спутники ГЛОНАСС'))
                for i in sats_glon:
                    fig.add_trace(go.Scatter(x=[j for j in range(int(len((self.file_obs) / 46)))],
                                             y=self.take_parametr('snrSat', sat_number=i,
                                                                  sat_type='glon',
                                                                  data_type='full'),
                                             name='№' + str(i)),
                                  row=1, col=1)

                fig.update_layout(height=900, width=1900,
                                  title_text="СШ спутников")
                fig.update_xaxes(title_text="Номера отсчетов, 1/с")
                # fig.show()
                plot(fig)

            elif len(sats_gps) != 0:
                fig = make_subplots(rows=1, cols=1,
                                    subplot_titles=('Спутники GPS'))
                for i in sats_gps:
                    fig.add_trace(go.Scatter(x=[j for j in range(int(len((self.file_obs) / 46)))],
                                             y=self.take_parametr('snrSat', sat_number=i,
                                                                  sat_type='gps',
                                                                  data_type='full'),
                                             name='№' + str(i)),
                                  row=1, col=1)

                fig.update_layout(height=900, width=1900,
                                  title_text="СШ спутников")
                fig.update_xaxes(title_text="Номера отсчетов, 1/с")
                # fig.show()
                plot(fig)
            else:
                print('error, no sats')
                return 0

        else:  # PNAP
            if len(sats_glon) != 0 and len(sats_gps) != 0 and (len(sats_bdu) != 0 or len(sats_gal) != 0):

                fig = make_subplots(rows=3, cols=1,
                                    vertical_spacing=0.1,
                                    # shared_xaxes=True,
                                    subplot_titles=('Спутники ГЛОНАСС',
                                                    'Спутники GPS',
                                                    'Спутники BDU, GAL'))
                for i in sats_glon:
                    fig.add_trace(go.Scatter(x=self.take_parametr('timeSat',
                                                                  sat_number=i,
                                                                  sat_type='glon'),
                                             y=self.take_parametr('snrSat',
                                                                  sat_number=i,
                                                                  sat_type='glon'),
                                             name='№' + str(i)),
                                  row=1, col=1)
                for i in sats_gps:
                    fig.add_trace(go.Scatter(x=self.take_parametr('timeSat',
                                                                  sat_number=i,
                                                                  sat_type='gps'),
                                             y=self.take_parametr('snrSat', 
                                                                  sat_number=i, 
                                                                  sat_type='gps'),
                                             name='№' + str(i)),
                                  row=2, col=1)
                for i in sats_gal:
                    fig.add_trace(go.Scatter(x=self.take_parametr('timeSat',
                                                                  sat_number=i,
                                                                  sat_type='gal'),
                                             y=self.take_parametr('snrSat',
                                                                  sat_number=i,
                                                                  sat_type='gal'),
                                             name='№' + str(i) + ' bdu'),
                                  row=3, col=1)
                for i in sats_bdu:
                    fig.add_trace(go.Scatter(x=self.take_parametr('timeSat',
                                                                  sat_number=i,
                                                                  sat_type='bdu'),
                                             y=self.take_parametr('snrSat',
                                                                  sat_number=i,
                                                                  sat_type='bdu'),
                                             name='№' + str(i) + ' gal'),
                                  row=3, col=1)

                fig.update_layout(height=900, width=1900,
                                  title_text="СШ спутников")
                fig.update_xaxes(title_text="Время, с")
                fig.show()
                # plot(fig)

    def accordance_of_hight(self, sl_l=0, sl_r=0, data_corr=False,
                            in_nav_corr=False, xaxe='num'):
        # Функция для вывода графика сопоставлений по координатам/скоростям с графиком ошибок
        '''
        '''
        if data_corr:
            self.data_correction = True
        if in_nav_corr:
            self.in_nav_correction = True

        gdop = self.take_parametr('gdop')
        # dflag = self.take_parametr('decisionFlag')
        i_start = 0
        i_end = len(gdop)

        i_start += sl_l
        i_end -= sl_r
        # print(i_start, i_end,'I')

        if xaxe == 'num':
            xaxe = self.take_parametr(self.num)
            if self.prot == 1:
                xaxe_name = "Номера отсчетов, 1/с"
            else:
                xaxe_name = "Номера отсчетов, 1/10с"
        elif xaxe == 'time':
            xaxe = tf.sec_to_time(self.take_parametr('time').to_numpy())
            xaxe_name = "Время"
        elif xaxe == 'sec':
            xaxe = self.take_parametr('time').to_numpy()
            xaxe_name = "Время, с"
        else:
            print('incorrect input xaxe')
            return 0

        dif_full = self.difference_by('coor', show_graphs=False)

        blh = mf.xyz_blh(self.take_parametr('userX()'),
                         self.take_parametr('userY()'),
                         self.take_parametr('userZ()'))

        hight = blh[2]

        dif_hight = dif_full[5]

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.04,
                            subplot_titles=('Высота, м', 'Ошибка по высоте, м'))

        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=hight[i_start:i_end],
                                 name=''),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=xaxe[i_start:i_end],
                                 y=dif_hight[i_start:i_end],
                                 name=''),
                      row=2, col=1)
        fig.update_layout(height=900, width=1900,
                          title_text="График соответствия высоты и ошибки по высоте")
        fig.update_xaxes(title_text=xaxe_name, col=1, row=2)
        # fig.show()
        plot(fig)

        return 1
