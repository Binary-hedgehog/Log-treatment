from math import atan, pi, sin, cos, sqrt, acos, asin, acosh, asinh, cosh, sinh

import numpy as np

# МОЯ ЛИЧНАЯ МАЛЕНЬКАЯ БИБЛИОТЕКА В КОТОРОЙ ЛЕЖАТ ВСЕ РЕАЛИЗОВАННЫЕ ФУНКЦИИ


# ИСХОДНЫЕ ДАННЫЕ ДЛЯ WGS-84
c_speed = 299792458  # СКОРОСТЬ СВЕТЫ
eks = 0.00669437999014
a = 6378137
b = 6356752.3142
alf = (a-b)/a
eks2 = eks/(1-eks)
###структура списков с координатами list = [b,l,h,x,y,z]###
compas_point =      [55.739776, 37.787815, 187.376, 2844181.913, 2205204.968, 5248340.243]
compas_point_op14 = [55.740273, 37.788794, 184.963, 2844107.395708284, 2205225.103510066, 5248370.201976600]
tatarskaya_point =  [55.736411, 37.636278, 185, 2850248.945160372, 2197863.567382474, 5248128.113947969]
svmc1_point =       [55.5366, 38.1772, 128.646, 2843783.41532259, 2236005.047984719, 5235524.735569037]
svmc2_point =       [55.53907, 38.17202, 128.711, 2843807.334413563, 2235607.8317871485, 5235680.400647452]
onp2_point =        [55.53728, 38.17549, 131.331, 2843802.272020579, 2235882.532438007, 5235569.790714532]
syr_idlib_point =   [35.9332921390998, 36.63599744144252, 442, 4149193.679766917, 3085509.6923088073, 3722460.350054406]
syr_hmeimim_point = [35.41371890952971, 35.94863134703012, 48, 4212867.908991824, 3055060.628206721, 3675398.247658029]



def spline(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, x):  # ИНТЕРПОЛЯЦИЯ КУБИЧЕСКИМ СПЛАЙНОМ

    a0 = y0
    a1 = y1
    a2 = y2
    a3 = y3
    a4 = y4

    h1 = x1 - x0
    h2 = x2 - x1
    h3 = x3 - x2
    h4 = x4 - x3

    k1 = 3 * ((a2 - a1) / h2 - (a1 - a0) / h1)
    k2 = 3 * ((a3 - a2) / h3 - (a2 - a1) / h2)
    k3 = 3 * ((a4 - a3) / h4 - (a3 - a2) / h3)

    c2 = (k2 - k1 * h2 / (2 * h1 + 2 * h2) - k3 * h3 / (2 * h3 + 2 * h4)) / (2 * h2 + 2 * h3 - h2**2 / (2 * h1 + 2 * h2) - h2 * h3 / (2 * h3 + 2 * h4))
    c1 = (k1 - c2 * h2) / (2 * h1 + 2 * h2)
    c3 = (k3 - c2 * h2) / (2 * h3 + 2 * h4)

    d1 = c1 / (3 * h1)
    d2 = (c2 - c1) / (3 * h2)
    d3 = (c3 - c2) / (3 * h3)

    b1 = (a1 - a0) / h1 + 2 * c1 * h1 / 3
    b2 = (a2 - a1) / h2 + (2 * c2 - c1) * h2 / 3
    b3 = (a3 - a2) / h3 + (2 * c3 - c2) * h3 / 3

    if x3 <= x:
        y = a3 + b3 * (x - x3) + c3 * (x - x3)**2 + d3 * (x - x3)**3
    elif x2 <= x < x3:
        y = a2 + b2 * (x - x2) + c2 * (x - x2)**2 + d2 * (x - x2)**3
    elif x < x2:
        y = a1 + b1 * (x - x1) + c1 * (x - x1)**2 + d1 * (x - x1)**3
    return y


def linar(x0, x1, y0, y1, x):  # ЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ
    return y0 + (x-x0)*(y1-y0)/(x1-x0)


def N1(B):  # КРИВИЗНА ПЕРВОГО ВЕРТИКАЛА 
    return a / ((1 - eks * np.sin(B)**2)**(1/2))


def atan_deg(x):  # ФУНКЦИЯ ВЫВОДА АРКТАНГЕНС В ГРАДУСАХ
    return atan(x) * 180 / pi


def xyz_blh(x, y, z):  # Функция преобразования геоцентрических координат в геодезические (иттеративаная)
    # Выходные данные в градусах
    q = (x**2 + y**2) ** (1/2)
    b1 = np.arctan(z / (q - q * eks))
    h1 = q / np.cos(b1) - N1(b1)
    b2 = np.arctan(z / (q - q * N1(b1) * eks / (N1(b1) + h1)))
    h2 = q / np.cos(b2) - N1(b2)
    b3 = np.arctan(z / (q - q * N1(b2) * eks / (N1(b2) + h2)))
    h3 = q / np.cos(b3) - N1(b3)
    b = b3 * 180 / pi
    h = h3
    L = np.arctan(y / x) * 180 / pi
    return b, L, h


def blh_xyz(B, L, H):  # Функция преобразования геодезических координат в геоцентрические
    # Данные на вход в градусах
    B = B * pi / 180  # преобразование в радианы
    L = L * pi / 180
    x = (N1(B) + H) * cos(B) * cos(L)
    y = (N1(B) + H) * cos(B) * sin(L)
    z = (N1(B) + H - eks * N1(B)) * sin(B)
    return x, y, z


def mat_ozh(x) -> float:  # МАТЕМАТИЧЕСКОЕ ОЖИДАНИЕ ДЛЯ МАССИВОВ ndarray
    return sum(x) / len(x)


def disp(x) -> float:  # ДИСПЕРИСИЯ ДЛЯ МАССИВОВ ndarray
    return mat_ozh((x - mat_ozh(x))**2)


def sko(x) -> float:  # СКО ДЛЯ МАССИВОВ ndarray
    return sqrt(disp(x))


def mat_ozh_list(x: list) -> float:  # МАТЕМАТИЧЕСКОЕ ОЖИДАНИЕ ДЛЯ СПИСКОВ list
    m = 0
    for i in x:
        m = m + i
    return m/len(x)


def disp_list(x: list) -> float:  # ДИСПЕРИСИЯ ДЛЯ СПИСКОВ list
    d = 0
    for i in x:
        d = d + (i - mat_ozh_list(x))**2
    return d / len(x)


def sko_list(x: list) -> float:  # СКО ДЛЯ СПИСКОВ list
    return sqrt(disp_list(x))


def oshibka(x, e=2):  # Рассчет ошибки m + 2 (3) сигма
    o = abs(mat_ozh(x)) + abs(sko(x)*e)
    print(mat_ozh(x), '-- математическое ожидание')
    print(sko(x), '-- СКО')
    if e == 2:
        a = ' 95%'
    elif e == 3:
        a = ' 99%'
    else:
        print('incorrect input "e"')
        return 0
    print(o, '-- ошибка' + a)
    return o


def sign(x):  # ОПРЕДЕЛЕНИЕ ЗНАКА ЧИСЛА
    if x > 0:
        return 1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def sr_kv(x, y, z):  # КОРЕНЬ СУММЫ КВАДРАТОВ
    return sqrt(x**2 + y**2 + z**2)


def kub(a, b, c, d):  # НАХОЖДЕНИЕ КОРНЕЙ УРАВНЕНИЯ 3Й СТЕПЕНИ
    a_1 = b/a
    b_1 = c/a
    c_1 = d/a
    # РЕШЕНИЕ МЕТОДОМ ВИЕТО-КАРДАНО
    q = (a_1**2 - 3 * b_1) / 9
    r = (2 * a_1**3 - 9 * a_1*b_1 + 27 * c_1) / 54

    if r**2 >= q**3:
        if q > 0:
            t = acosh(abs(r) / sqrt(q**3)) / 3
            x = -2 * sign(r) * sqrt(q) * cosh(t) - a_1 / 3
        elif q < 0:
            t = asinh(abs(r) / sqrt(abs(q)**3)) / 3
            x = -2 * sign(r) * sqrt(abs(q)) * sinh(t) - a_1 / 3
        # print('odin koren')
        # print(x)
        return x

    elif r**2 == q**3:

        x_1 = -2 * r**(1/3) - a_1 / 3
        x_2 = r**(1/3) - a_1 / 3
        # print('dva корня')
        # print(x_1,x_2)
        return x_1, x_2

    else:
        t = acos(r / sqrt(q**3)) / 3
        x_1 = -2 * sqrt(q) * cos(t) - a_1 / 3
        x_2 = -2 * sqrt(q) * cos(t + (2 * pi / 3)) - a_1 / 3
        x_3 = -2 * sqrt(q) * cos(t - (2 * pi / 3)) - a_1 / 3
        # print('три корня')
        # print(x_1,x_2,x_3)
        return x_1, x_2, x_3


def spline_v3(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, x):  # Cплайн интерполяция

    a1 = y0
    a2 = y1
    a3 = y2
    a4 = y3

    h1 = x1 - x0
    h2 = x2 - x1
    h3 = x3 - x2
    h4 = x4 - x3
    k1 = 3 * ((y2 - y1) / h2 - (y1 - y0) / h1)
    k2 = 3 * ((y3 - y2) / h3 - (y2 - y1) / h2)
    k3 = 3 * ((y4 - y3) / h4 - (y3 - y2) / h3)

    c3 = (k2 - k1 * h2 / (2 * h2 + 2 * h1) - k3 * h3 / (2 * h3  + 2 * h4)) / (2 * h2 + 2 * h3 - h2**2 / ( 2 * h1 + 2 * h2) - h3**2 / (2 * h3  + 2 * h4))
    c2 = (k2 - c3 * h2) / (2 * h2 + 2 * h1)
    c4 = (k3 - c3 * h3) / (2 * h3 + 2 * h4)

    d1 = c2 / (3 * h1)
    d2 = (c3 - c2) / (3 * h2)
    d3 = (c4 - c3) / (3 * h3)
    d4 = -c4 / (3 * h4)

    b1 = (y1 - y0) / h1 - c2 * h2 / 3
    b2 = (y2 - y1) / h2 - (2 * c2 + c3) * h2 / 3
    b3 = (y3 - y2) / h3 - (2 * c3 + c4) * h3 / 3
    b4 = (y4 - y3) / h4 - 2 * c4 * h4 / 3

    if x3 <= x:
        y = a4 + b4 * (x - x3) + c4 * (x - x3)**2 + d4 * (x - x3)**3
    elif x2 <= x < x3:
        y = a3 + b3 * (x - x2) + c3 * (x - x2)**2 + d3 * (x - x2)**3
    elif x1 <= x < x2:
        y = a2 + b2 * (x - x1) + c2 * (x - x1)**2 + d2 * (x - x1)**3
    elif x < x1:
        y = a1 + b1 * (x - x0) + d1 * (x - x0)**3
    return y


def koef_glon():  # Коэффициент для ГЛОНАСС СТ
    return 1e6 / (2048 * 511) * 1e-9 * c_speed


def koef_gps():  # Коэффициент для GPS
    return 1e6 / (1023 * 512) * 1e-9 * c_speed


def koef_vt():  # Коэффициент для ГЛОНАСС ВТ
    return 1e6 / (5110 * 256) * 1e-9 * c_speed


def approx(vvod, time, time0, F0="0"):  # ФУНКЦИЯ АППРОКСИМАЦИИ ВХОДНЫХ ДАННЫХ (вообще-то это интерполяция...)
    '''
    vvod - координаты имитатора!
    time - время работы прибора
    time0 - время сценария имитатора
    '''
    # print('vvod',vvod)
    # print('time',time)
    # print('time_0',time0)
    vivod = np.zeros_like(time)
    a = 0
    ii = 0
    for i in time:
        m = 1
        j = a

        while j < len(time0) - 3:
            k = abs(i-time0[j])
            k1 = abs(i-time0[j+1])

            if k < m:
                # print(k)
                a = j
                m = k

            j += 1
            if k1 > k and k >= m:
                break

        if m == 1:
            continue

        vivod[ii] = spline(time0[a-2], time0[a-1], time0[a], time0[a+1],
                           time0[a+2], vvod[a-2], vvod[a-1], vvod[a],
                           vvod[a+1], vvod[a+2], i)
        # print((time0[a-2],time0[a-1],time0[a],time0[a+1],time0[a+2],vvod[a-2],vvod[a-1],vvod[a],vvod[a+1],vvod[a+2],i))
        # print(vivod[ii])
        if F0 != "0":
            vivod[ii] += F0[np.where(time == i)[0]]
            # print(np.where(time == i)[0])
        ii += 1

    return vivod


def approx_linar(vvod, time, time0):  # ФУНКЦИЯ АППРОКСИМАЦИИ ВХОДНЫХ ДАННЫХ Линейной аппроксимацией
    '''
    vvod - координаты имитатора!
    time - время работы прибора
    time0- время сценария имитатора
    '''
    # print(len(time0))
    # print('vvod',vvod)
    # print('time',time)
    # print('time_0',time0)
    vivod = np.zeros_like(time)
    a = 0
    ii = 0
    for i in time:
        m = 1
        j = a
        while j < len(time0) - 1:
            k = abs(i-time0[j])
            k1 = abs(i-time0[j+1])
            if k < m:
                # print(k)
                a = j
                m = k

            j += 1
            if k1 > k and k >= m:
                break

        if m == 1:
            continue

        # vivod[ii] = spline(time0[a-2],time0[a-1],time0[a],time0[a+1],time0[a+2],vvod[a-2],vvod[a-1],vvod[a],vvod[a+1],vvod[a+2],i)
        vivod[ii] = linar(time0[a], time0[a+1], vvod[a], vvod[a+1], i)
        # print((time0[a-2],time0[a-1],time0[a],time0[a+1],time0[a+2],vvod[a-2],vvod[a-1],vvod[a],vvod[a+1],vvod[a+2],i))
        # print(vivod[ii])
        ii += 1
    return vivod


def psevdorange(sat_range, time, time0, time_track, T0):  # ФУНКЦИЯ АППРОКСИМАЦИИ ДАЛЬНОСТЕЙ ИМИТАТОРА, И ВЫЧИТАНИЕ clock_drift (T0)

    vivod = np.zeros_like(time)

    a = 0
    ii = 0
    for i in time:

        m = 1
        j = a
        while j < len(time0) - 2:

            k = abs(i-time0[j])
            k1 = abs(i-time0[j+1])

            if k < m:
                a = j
                m = k

            j += 1
            if k1 > k and k >= m:
                break

        if m == 1:
            continue

        vivod[ii] = spline(time0[a - 2], time0[a - 1], time0[a], time0[a + 1],
                           time0[a + 2], sat_range[a - 2], sat_range[a - 1],
                            sat_range[a], sat_range[a + 1], sat_range[a + 2], i)
        vivod[ii] += T0[np.where(time_track == i)[0]]
        ii += 1

    return vivod


def approx3(vvod1, vvod2, vvod3, time, time0):  # ФУНКЦИЯ АППРОКСИМАЦИИ ВХОДНЫХ ДАННЫХ ДЛЯ ТРЕХ ЭЛЕМЕНТОВ
    '''
    vvod1,2,3 - координаты имитатора!
    time - время работы прибора
    time0- время сценария имитатора
    '''
    # print(vvod)
    # print(vvod[2342])
    # vivod = np.empty_like(time)
    # print(len(time),len(time0),'len')

    vivod1 = np.zeros_like(time)
    vivod2 = np.zeros_like(time)
    vivod3 = np.zeros_like(time)
    a = 0
    ii = 0
    for i in time:

        m = 1
        j = a
        while j < len(time0)-3:

            k = abs(i-time0[j])
            k1 = abs(i-time0[j+1])
            # print(k,"k")

            if k < m:
                a = j
                m = k
            j += 1
            if k1 > k and k >= m:
                break

        if m == 1:
            continue

        # print(vvod[a-2],vvod[a-1],vvod[a],vvod[a+1],vvod[a+2])
        vivod1[ii] = spline(
            time0[a - 2],
            time0[a - 1],
            time0[a],
            time0[a + 1],
            time0[a + 2],
            vvod1[a - 2],
            vvod1[a - 1],
            vvod1[a],
            vvod1[a + 1],
            vvod1[a + 2],
            i)
        vivod2[ii] = spline(
            time0[a - 2],
            time0[a - 1],
            time0[a],
            time0[a + 1],
            time0[a + 2],
            vvod2[a - 2],
            vvod2[a - 1],
            vvod2[a],
            vvod2[a + 1],
            vvod2[a + 2],
            i)
        vivod3[ii] = spline(
            time0[a - 2],
            time0[a - 1],
            time0[a],
            time0[a + 1],
            time0[a + 2],
            vvod3[a - 2],
            vvod3[a - 1],
            vvod3[a],
            vvod3[a + 1],
            vvod3[a + 2],
            i)
        # print(vivod[ii]," ",ii)
        ii += 1
    # print(vivod,len(vivod),"vivod")
    return vivod1, vivod2, vivod3


def lla2matrixT(lat, lon):  # Матричные игры для функции ниже
    T = np.zeros((3, 3))
    T[0][0] = -np.sin(lat)*np.cos(lon)
    T[0][1] = -np.sin(lat)*np.sin(lon)
    T[0][2] = np.cos(lat)
    T[1][0] = -np.sin(lon)
    T[1][1] = np.cos(lon)
    T[1][2] = 0
    T[2][0] = np.cos(lat)*np.cos(lon)
    T[2][1] = np.cos(lat)*np.sin(lon)
    T[2][2] = np.sin(lat)
    return T


def lla2xyz(vx, vy, vz, lat, lon):  # ПЕРЕВОД СКОРОСТЕЙ ИЗ XYZ B BLH
    lat = lat * pi / 180
    lon = lon * pi / 180
    T = lla2matrixT(lat, lon)
    vlla = np.zeros(3)
    T = np.linalg.pinv(T)
    for i in range(3):
        vlla[i] = T[0][i]*vx + T[1][i]*vy + T[2][i]*vz
    return vlla


def lla2xyz_mas(vx, vy, vz, lat, lon):  # ПЕРЕВОД СКОРОСТЕЙ ИЗ XYZ B BLH МАССИВОМ ndarray (NUMPY)

    vb = np.zeros_like(vx)
    vl = np.zeros_like(vx)
    vh = np.zeros_like(vx)
    for i in range(len(vx)):
        # print(vx[i],vy[i],vz[i],lat[i],lon[i])
        ccc = lla2xyz(vx[i], vy[i], vz[i], lat[i], lon[i])
        vb[i] = ccc[0]
        vl[i] = ccc[1]
        vh[i] = ccc[2]
    return vb, vl, vh


def exp_moving_average(x, prev, alf=0.5):  # функция скользящего среднего
    ema = alf*x + (1-alf)*prev
    return ema


def e_m_a(x, alf=0.1):  # скользящее среднее над массивом
    a = x[0]
    for i in range(1, len(x)):
        x[i] = exp_moving_average(x[i], a, alf)
        a = x[i]
    return x


def slice_r(self, x, n=1):  # уменьшение длины массива на N справа
    return x[0:len(x) - n]


def slice_l(self, x, n=1):  # уменьшение длины массива на N слева
    return x[n:len(x)]


def from_bin_to_dec(x, val=1):  # перевод из двоичного в десятичный с домножением на значение младшего бита
    # print(x)
    y = 0
    for i in range(len(x)):
        y += int(x[-(i+1)]) * val * 2**(i)
        # print(y)
    return y


def extrop_coor_with_vel(coor, vel, dt=1):  # Экстрополяция координат скоростью
    proection = []
    for i in range(len(coor)):
        if i == 0:
            proection.append(coor[i])
        else:
            proection.append(coor[i-1]+vel[i-1]*dt)
    return proection


def l_shift(x, n):  # Сдвиг массива влево, на Н
    y = np.zeros_like(x)
    for i in range(len(x)):
        if i+n < len(x):
            y[i] = x[i+n]
        else:
            y[i] = x[len(x)-1]
    return y


def r_shift(x, n):  # Сдвиг массива вправо, на Н
    y = np.zeros_like(x)
    for i in range(len(x)):
        if i < n:
            y[i] = x[0]
        else:
            y[i] = x[i-n]
    return y


def angle_between(x0, y0, z0, x1, y1, z1):  # Угол между двумя точками в пространтсве
    # На вход градусы, на выход градусы
    x0 = x0/180*pi
    y0 = y0/180*pi
    x1 = x1/180*pi
    y1 = y1/180*pi
    angle = np.arctan((z1-z0)/np.sqrt((x1-x0)**2+(y1-y0)**2))
    return angle*180/pi


def deg_conv(deg_, min_, sec_) -> float:  # Конвертирование градусов в десятичный формат
    return deg_+min_/60+sec_/3600
