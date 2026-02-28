import cv2
import numpy as np
import os

# ===== НАСТРОЙКИ =====
# Путь к папке с калибровочными изображениями (распакованными!)
calib_dir = r"C:\Users\Алексей\Downloads\UFA\calib"
square_size_mm = 10                    # размер клетки в миллиметрах (10 мм = 1 см)
pattern_size = (4, 7)                    # внутренние углы: 4 по горизонтали, 7 по вертикали
# =====================

# Собираем все изображения шахматки в папке
image_paths = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_paths:
    print(f"Ошибка: в папке {calib_dir} нет изображений.")
    exit()

pixels_per_cm_list = []

for img_path in image_paths:
    print(f"Обрабатывается: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("  Не удалось прочитать файл, пропускаем.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        print("  Не удалось найти шахматную доску. Пропускаем.")
        continue

    # Уточняем координаты углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Рисуем углы для визуализации (опционально)
    cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
    cv2.imshow("Chessboard corners", img)
    cv2.waitKey(500)  # показываем 0.5 секунды
    cv2.destroyAllWindows()

    # Вычисляем расстояние между первым и последним углом по горизонтали
    # corners2[0] - левый верхний, corners2[pattern_size[0]-1] - правый верхний
    pixel_dist = np.linalg.norm(corners2[0] - corners2[pattern_size[0]-1])
    real_dist_mm = (pattern_size[0] - 1) * square_size_mm
    pixels_per_mm = pixel_dist / real_dist_mm
    pixels_per_cm = pixels_per_mm * 10

    pixels_per_cm_list.append(pixels_per_cm)
    print(f"  Результат: {pixels_per_cm:.3f} px/см")

if pixels_per_cm_list:
    avg = np.mean(pixels_per_cm_list)
    print(f"\nСредний масштаб по {len(pixels_per_cm_list)} изображениям: {avg:.3f} px/см")
    with open("calibration_result.txt", "w") as f:
        f.write(str(avg))
    print("Коэффициент сохранён в calibration_result.txt")
else:
    print("Не удалось обработать ни одного изображения.")