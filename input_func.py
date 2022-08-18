import numpy as np


def input_xyz():  # Функция возращает данные имитатора в виде массива
    reading = open('IM-2/f_xyz.txt', 'r')
    writing = open('IM-2/f_xyz_.txt', 'w')
    i = 0
    for line in reading:
        if i == 0:
            i += 1
            continue
        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('IM-2/f_xyz_.txt')


def input_blh():  # Функция возращает данные имитатора в виде массива
    reading = open('IM-2/f_blh.txt', 'r')
    writing = open('IM-2/f_blh_.txt', 'w')
    i = 0
    for line in reading:
        if i == 0:
            i += 1
            continue
        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('IM-2/f_blh_.txt')


def input_frg():  # Функция возращает данные имитатора в виде массива
    reading = open('IM-2/frg.txt', 'r')
    writing = open('IM-2/frg_.txt', 'w')
    for line in reading:

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('IM-2/frg_.txt')


def input_frn():  # РЕДАКТИРОВАНИЕ ФАЙЛА ДЛЯ ЧИТАЕМОСТИ ПРОГРАММОЙ
    reading = open('IM-2/frn.txt', 'r')
    writing = open('IM-2/frn_.txt', 'w')
    for line in reading:

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('IM-2/frn_.txt')


def input_fvg():  # Функция возращает данные имитатора в виде массива

    reading = open('IM-2/fvg.txt', 'r')
    writing = open('IM-2/fvg_.txt', 'w')
    for line in reading:

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('IM-2/fvg_.txt')


def input_fvn():  # Функция возращает данные имитатора в виде массива

    reading = open('IM-2/fvn.txt', 'r')
    writing = open('IM-2/fvn_.txt', 'w')
    for line in reading:

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('IM-2/fvn_.txt')


def input_track_rinex():  # Функция возвращает массив навигационных параметров track ППА (rinex)
    reading = open('logs/rinex/track_0.txt', 'r')
    writing = open('logs/rinex/track.txt', 'w')
    u = 0
    for line in reading:
        if u == 0:
            u = 1
            continue

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('logs/rinex/track.txt')


def correct_obs_rinex():  # Функция конвертации файла obs для последующей обработки
    reading = open('logs/rinex/obs_0.txt', 'r')
    writing = open('logs/rinex/obs.txt', 'w')

    i = 0
    j = 0
    for line in reading:

        if i == 4 and j == 0:
            writing.write(line)
            j = -1

        if i < 5:
            i += 1

        elif 5 <= i < 37:
            line = line.replace(',', '.')
            line = line.replace('---', '0')
            writing.write(line)
            i += 1

        elif 37 <= i < 40:
            i += 1

        elif 40 <= i:
            line = line.replace('---', '0')
            line = line.replace(',', '.')
            writing.write(line)
            i += 1
            if i >= 54:
                i = 0

    reading.close()
    writing.close()


def input_kbti_nav(name=''):  # Функция возвращает массив данных КБТИ навигация
    if name == '':
        name = 'logs/kbti/ППА_навигация.txt'
    reading = open(name, 'r')
    writing = open(name+'_', 'w')
    u = 0
    for line in reading:
        if u < 2:
            u += 1
            continue
        writing.write(line)

    reading.close()
    writing.close()
    return np.genfromtxt(name+'_')


def input_kbti_land(name=''):  # Функция возвращает массив данных КБТИ посадка
    if name == '':
        name = 'logs/kbti/ППА_посадка.txt'
    reading = open(name, 'r')
    writing = open(name+'_', 'w')
    u = 0
    for line in reading:
        if u < 2:
            u += 1
            continue
        writing.write(line)

    reading.close()
    writing.close()
    return np.genfromtxt(name+'_')


def input_kbti_et():  # Функция возвращает массив данных КБТИ эталон

    reading = open('logs/kbti/Эталон_КБТИ.txt', 'r')
    writing = open('logs/kbti/Эталон_КБТИ_.txt', 'w')
    u = 0
    for line in reading:
        if u < 2:
            u += 1
            continue
        writing.write(line)

    reading.close()
    writing.close()
    return np.genfromtxt('logs/kbti/Эталон_КБТИ_.txt')


def input_track_PNAP():  # Функция возвращает массив навигационных параметров track ПНАП

    reading = open('logs/pnap/track_0.txt', 'r')
    writing = open('logs/pnap/track.txt', 'w')
    u = 0
    for line in reading:
        if u == 0:
            u = 1
            continue

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('logs/pnap/track.txt')


def input_state1():  # Функция возвращает массив навигационных параметров State1 ПНАП

    reading = open('logs/pnap/state1.txt', 'r')
    writing = open('logs/pnap/state1_.txt', 'w')
    u = 0
    for line in reading:
        if u == 0:
            u = 1
            continue

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('logs/pnap/state1_.txt')


def input_state2():  # Функция возвращает массив навигационных параметров State2 ПНАП

    reading = open('logs/pnap/state2.txt', 'r')
    writing = open('logs/pnap/state2_.txt', 'w')
    u = 0
    for line in reading:
        if u == 0:
            u = 1
            continue

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('logs/pnap/state2_.txt')


def input_obs_PNAP():  # Функция возвращает массив навигационных параметров obs ПНАП

    reading = open('logs/pnap/obs_0.txt', 'r')
    writing = open('logs/pnap/obs_0_.txt', 'w')
    u = 0
    for line in reading:
        if u == 0:
            u = 1
            continue

        line = line.replace(',', '.')
        writing.write(line)

    reading.close()
    writing.close()

    return np.genfromtxt('logs/pnap/obs_0_.txt')


def init_array():  # Инициализация пустого ndarray
    return np.zeros(1)


def init_dict():  # Инициализация пустого dict
    return dict.fromkeys(['None'])


def pnap_prepare_data(frame_track, frame_state1, frame_state2, frame_obs):  # Подготовка данных ПНАП для обработки
    # Обрезает по началу навигации, и вырезает моменты перезапуска блоков (кроме файла obs_0)
    frame_obs = frame_obs[frame_obs['timeSat'] != 0]

    frame_s1 = frame_state1[frame_state1['gdop'] <= 50]
    frame_s1 = frame_s1[frame_s1['gdop'] >= 1]

    st_time_st = frame_s1['time'].iloc[1]
    st_index_st = frame_s1['num_state1_bnm'].iloc[1]

    frame_tr0 = frame_track[frame_track['decisionFlag'] == 1]
    tr_time_st = frame_tr0['time'].iloc[0]

    if (tr_time_st - st_time_st) > 5:
        frame_s1 = frame_s1[frame_s1['time'] >= tr_time_st]
        st_time_st = frame_s1['time'].iloc[1]
        st_index_st = frame_s1['num_state1_bnm'].iloc[1]

    frame_tr = frame_track[frame_track['time'] == st_time_st]
    st_index_tr = int(frame_tr['num_track_bnm'])
    print(st_index_tr, st_index_st, 'start of track and state')

    end_index_st = 0

    t0 = frame_state1['time'].iloc[st_index_st]
    frame1 = frame_state1['time'].iloc[(st_index_st+1):]
    for i in frame1:
        if i < t0:
            if i < 10:
                frame0 = frame_state1[frame_state1['time'] == t0]
                end_index_st = int(frame0['num_state1_bnm'])
                break
            #t0 = i
        else:
            t0 = i

    if end_index_st == 0:
        time1 = frame_state1['time'].iloc[-1]
        frame_tmp = frame_state1[frame_state1['time'] == time1]
        if len(frame_tmp) > 1:
            tmp = frame_tmp['num_state1_bnm'].iloc[0]
            frame_state1 = frame_state1.iloc[:tmp]
            frame_state2 = frame_state2.iloc[:tmp]

        frame_tr = frame_track[frame_track['time'] == time1]
        end_index_tr = frame_tr['num_track_bnm'].iloc[0]

        print(end_index_tr, 'end of track')
        return (frame_track.iloc[st_index_tr:end_index_tr],
                frame_state1.iloc[st_index_st:],
                frame_state2.iloc[st_index_st:],
                frame_obs)

    else:
        frame_tr = frame_track[frame_track['time'] == t0]
        end_index_tr = int(frame_tr['num_track_bnm'])
        print("ATTENTION")
        print("Y've got two(or more) inclusion on this log!")
        print("Obs_0.txt was't correct, do not use this data")
        print("TY FOR YR ATTENTION")
        print(end_index_tr, end_index_st, 'end of track and state')
        return (frame_track.iloc[st_index_tr:end_index_tr],
                frame_state1.iloc[st_index_st:end_index_st],
                frame_state2.iloc[st_index_st:end_index_st],
                frame_obs)


def rinex_prepare_data(frame_track, frame_obs):  # Подготовка данных ППА для обработки
    # Пока без обработки файла obs
    frame0 = frame_track[frame_track['gdop'] >= 1]
    frame0 = frame0[frame0['gdop'] <= 10]
    frame0 = frame0[frame0['decisionFlag'] == 1]
    frame0 = frame0[frame0['time'] != 0.0]
    st_index_tr = frame0['num_track'].iloc[1]
    # print(st_index_tr,'start index track')
    frame_track = frame_track.iloc[st_index_tr:]

    frame_track = frame_track[frame_track['decisionFlag'] <= 1]  # cut data by logic (dflag could be only 0\1)
    frame_track = frame_track[frame_track['decisionFlag'] >= 0]
    return frame_track, frame_obs


def command_input(word: str) -> str:
    command = input()
    if word == 'command':
        if command in ['1', '2', '3', '4', '5', '6', '7', '8']:
            return command
        else:
            print('Неправильно введена команда, попробуйте еще раз')
            command_input('command')
    elif word == 'prot':
        if command in ['rinex', 'pnap']:
            return command
        else:
            print('Неправильно введен протокол, попробуйте еще раз')
            command_input('prot')
    elif word == 'diff':
        if command in ['coor', 'rate']:
            return command
        else:
            print('Неправильно введен параметр, попробуйте еще раз')
            command_input('diff')


def blh_input() -> list:
    print('Введите координаты точки\n'+'Широта=')
    B = input()
    print('Долгота=')
    L = input()
    print('Высота=')
    H = input()
    if len(B) == 0 or len(L) == 0 or len(H) == 0:
        print('Некорретный ввод координат (пустое значение)\n Попробуйте еще раз')
        blh_input()
    try:
        B = float(B)
        L = float(L)
        H = float(H)
    except ValueError:
        print('Только числа\n Попробуйте еще раз')
        blh_input()
    return B, L, H
###
