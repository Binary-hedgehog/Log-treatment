import numpy as np


def time_cheking(t0, t):  # ПРИВОД К ОБЩЕМУ ВРЕМЕНИ ГЛОНАСС И GPS
    # в данный момент не применяется
    if abs(max(t0)-max(t)) >= 3600:
        t = t - 10800
        print('different time', max(t0), max(t), '- t0,t')
    else:
        print('similar time', max(t0), max(t))
    return t


def time_chek(t0, t):  # ПРИВОД К ОБЩЕМУ ВРЕМЕНИ ГЛОНАСС И GPS (БОЛЕЕ СЛОЖНОЕ УСЛОВИЕ)
    ii = 0
    '''
    t0 - время имитатора
    t  - время приемника
    '''
    # max_t = max(t)
    # max_t0 = max(t0)
    # min_t = t[0]
    # min_t0 = t0[0]
    if max(t)-max(t0) >= 3600 and t[0]-t0[0] >= 3600:
        t -= 10800
        return t
    elif max(t0)-max(t) >= 3600 and t0[0]-t[0] >= 3600:
        t += 10800
        return t
    elif 0:     # strange part of my dream
        if max(t)-max(t0) >= 3600 and t[0]-t0[0] <= 3600:
            for i in range(len(t)):
                if t[i] - t0[0+ii] <= 3600:
                    ii += 1
                    t[i] -= 17
                else:
                    break
            t[ii:len(t)] = t[ii:len(t)] - 10800
            print('different time, only gps was =', ii, '\\\\', max(t0), max(t), '- t0,t')
            return t
    else:
        # print('similar time')#,max(t0),max(t)
        return t


def time_remade(t):  # Функция которая приводит все время в логе к одной системе UTC или UTC +3
    # она теперь не нужна
    for i in range(len(t)-1):
        # if t[i+1] == 0.0 or :

        if t[i+1] - t[i] > 1:
            if t[i+1] - 10800 < t[i]:
                t[i+1] -= 10800  # + 2 ### и это что за нахуй то?
            else:
                t[i+1] -= 10800
        if t[i] - t[i+1] > 1:
            t[i+1] += 10800
    return t


def kbti_matching_time(time1, time2):  # Сравнение времени с КБТИ
    x = []
    y = []
    j = 0
    for i in range(len(time1)):
        while j < len(time2):
            if abs(time1[i] - time2[j]) > 0.09:
                x.append(i)
                y.append(j)
                j += 1
                break
            if time2[j] > time1[i]:
                break
            j += 1
    return x, y


def sec_to_time(time) -> list:  # Перевод секунд в часы, минуты, секунды
    t = []
    for i in time:
        h = i//3600
        m = i % 3660//60
        s = i % 3600 % 60
        s = str(s)
        t.append(str(h)+'.'+str(m)+'.'+s[0:2])
    return t


def search_time_swaps(arr):  # Функция поиска скачков времени в файле
    '''
    На вход подается массив(список) времени и смотрится количество нелогичных переходов
    Если таковых больше двух -> тут что-то не очень хорошо, надо быть внимательнее
    ...возможны множественные запуски изделия (А мы с таким не работаем)
    На выходе список индексов
    '''
    L = [indx + 1 for indx, val in enumerate(np.diff(arr)) if abs(val) > 3]

    if len(L) > 2:
        print('WARNING more than 2 ')
    return L


def pnap_prepare_time(frame_track, frame_state1):  # Функция преобразования времени ПНАПа к UTC, от аргумента типа времени
    '''
    без state2 и obs
    ...
    На входе может быть 1, 2 или 3 типа времени, при последнем случае одно из них внутреннее, а второе utc
    '''
    ''' f
    1 - привести весь файл к UTC
    2 - привести вторую часть файла к UTC
    3 - привести первую часть файла к UTC
    4 - привести середину к UTC
    '''
    type_times = set(frame_track['time_type'])
    indexes_tr = search_time_swaps(frame_track['time'])
    indexes_st = search_time_swaps(frame_state1['time'])

    if len(indexes_tr) != len(indexes_st):  # проверка на одинаковое количество переходов времени
        f_same = 0  # количество переходов не равно
    else:
        f_same = 1  # количество переходов равно

    if len(type_times) == 1:
        if 4 in type_times or 6 in type_times:  # либо уже utc либо не было навигации вообще, очень жаль
            return frame_track, frame_state1
        else:  # только другое время
            f = 1
    elif len(type_times) == 2:
        if 4 in type_times and 6 in type_times:  # внутреннее время и utc, все ок
            return frame_track, frame_state1
        elif 6 in type_times:  # внутреннее и другое время
            f = 2
        else:  # другое время и utc
            f = 3
    elif len(type_times) == 3:
        if 4 in type_times and 6 in type_times:  # внутреннее и еще одно и utc
            f = 4
        else:
            print('несколько включений, пока ошибка, ибо нефиг))')
            return 0
    else:
        print('Слишком много типов времени в файле, такого не бывает, разберись')
        return 0

    # выбор среза массива времени
    if f == 1:
        index1_st = 0
        index1_tr = 0
        index2_st = len(frame_state1['time']) - 1
        index2_tr = len(frame_track['time']) - 1
    elif f == 2:
        if f_same:
            index1_st = indexes_st[0]
        else:
            index1_st = 0
        index1_tr = indexes_tr[0]
        index2_st = len(frame_state1['time'])  # - 1
        index2_tr = len(frame_track['time'])  # - 1
    elif f == 3:    # if f == 3 and not f_same -> return frame_state1 (там только utc)
        index1_st = 0
        index1_tr = 0
        if f_same:
            index2_st = indexes_st[0]
        else:
            index2_st = -1
        index2_tr = indexes_tr[0]
    elif f == 4:
        index1_tr = indexes_st[0]
        index2_tr = indexes_tr[1]
        if f_same:
            index1_st = indexes_st[0]
            index2_st = indexes_st[1]
        else:
            # внутреннее есть, пропущено другое, есть utc
            if frame_state1.loc[indexes_st[0], 'time'] - frame_state1.loc[indexes_st[0]-1, 'time'] > 0:
                index1_st = 0
                index2_st = - 1
            else:  # пропущено внутреннее, есть другое и utc
                index1_st = 0
                index2_st = indexes_st[0]
    else:
        pass

    if 2 in type_times or 5 in type_times:  # glon time or msk time (it's same)
        frame_track.loc[index1_tr:index2_tr, 'time'] -= 10800
        if index2_st != -1:
            frame_state1.loc[index1_st:index2_st, 'time'] -= 10800
        return frame_track, frame_state1

    elif 0 in type_times:  # gps time
        time_tr = frame_track.loc[index1_tr:index2_tr-1, 'time'].to_numpy() % 86400
        if f not in [1, 2]:
            dt = round(time_tr[-1] - frame_track.loc[index2_tr, 'time'])
        else:
            dt = 18  # Костыль нужен точный год
        time_tr -= dt
        frame_track.loc[index1_tr:index2_tr-1, 'time'] = time_tr
        if index2_st != -1:
            time_st = frame_state1.loc[index1_st:index2_st-1, 'time'].to_numpy() % 86400
            time_st -= dt
            frame_state1.loc[index1_st:index2_st-1, 'time'] = time_st

        return frame_track, frame_state1

    elif 1 in type_times:  # bdu time
        pass  # same gps?
    elif 3 in type_times:  # gal time
        pass
    elif 7 in type_times or 6 in type_times:
        print('Smthng error in convert_time_to_utc')
        return 0

    return frame_track, frame_state1


def rinex_prepare_time(frame_track, frame_obs):  # Функция которая приводит все время в логе к одной системе UTC или UTC +3

    if 0 in set(frame_track['time_type']) and 3 in set(frame_track['time_type']):  # если время начинается с GPS
        indx1 = frame_track[frame_track['time_type'] == 0].index[0]
        t0_gps = frame_track.loc[indx1, 'time']  # значение времени по метке gps
        dt1 = round(t0_gps - frame_track.loc[indx1+1, 'time'])
        print(dt1, 'delta time gps_utc')
        frame_track.loc[indx1, 'time'] -= dt1
        # print(frame_track.loc[indx1,'time'])

        frame0 = frame_track[frame_track['decisionFlag'] == 1]
        frame0 = frame0.iloc[2:]  # свдвиг на 2 дабы на всякий случай

        # for obs
        indx_obs_1 = frame_obs[frame_obs['time'] == t0_gps].index[0]
        for i in range(46):
            if frame_obs.loc[indx_obs_1+i, 'time'] == t0_gps:
                frame_obs.loc[indx_obs_1+i, 'time'] -= dt1
        # end obs

        if 2 in set(frame0['time_type']):
            indx2 = frame0[frame0['time_type'] == 2].index[0]
            t0_glon = frame_track.loc[indx2, 'time']
            dt2 = round(frame_track.loc[indx2-1, 'time'] - t0_glon) + 10800
            print(dt2, 'error of delta time gps_glon')

            frame_track.loc[indx2:, 'time'] -= 10800
            frame_track.loc[indx1:indx2-1, 'time'] -= dt2
            # for obs
            indx_obs_2 = frame_obs[frame_obs['time'] == t0_glon].index[0]

            frame_obs.loc[indx_obs_1:indx_obs_2, 'time'] -= dt2
            frame_obs.loc[indx_obs_2:, 'time'] -= 10800
            # end
            return frame_track, frame_obs

        else:
            print('Only gps time_type')
            return frame_track, frame_obs

    else:
        print('Only glon time_type')
        return frame_track, frame_obs

