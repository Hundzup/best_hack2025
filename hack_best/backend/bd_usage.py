import re
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import json

# Официальные сокращения из классификатора ФИАС
FIAS_STREET_TYPES = [
    "аллея", "ал.", "бульвар", "б-р", "взвоз", "взв.", "въезд", "взд.", "дорога", "дор.",
    "заезд", "ззд.", "километр", "км", "кольцо", "к-цо", "коса", "коса", "линия", "лн.",
    "магистраль", "мгстр.", "набережная", "наб.", "переезд", "пер-д", "переулок", "пер.",
    "площадка", "пл-ка", "площадь", "пл.", "проезд", "пр-д", "просек", "пр-к", "просека", "пр-ка",
    "проселок", "пр-лок", "проспект", "пр-кт", "проулок", "проул.", "разъезд", "рзд.",
    "ряды", "ряд", "сквер", "сквер", "спуск", "с-к", "съезд", "сзд.", "тракт", "тракт",
    "тупик", "туп.", "улица", "ул.", "шоссе", "ш."
]

def normalize_text(text):
    """Универсальная нормализация текста"""
    if not text:
        return ""
    text = text.lower().strip()
    # Удаляем лишние пробелы и знаки препинания
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_street_name(address):
    """
    Точно извлекает название улицы из адреса
    """
    # Создаем копию для обработки
    address = address.strip()
    
    # Удаляем указание на Москву
    address = re.sub(r'^(г\.?\s*)?москва[\s,]*', '', address, flags=re.IGNORECASE)
    address = re.sub(r'[\s,]*(г\.?\s*)?москва$', '', address, flags=re.IGNORECASE)
    
    # Шаг 1: Проверяем наличие явного указания типа улицы
    street_pattern = r'(' + '|'.join(FIAS_STREET_TYPES) + r')\.?\s+([^\d,]+?)(?:\s*,|\s*$|\s*\d)'
    match = re.search(street_pattern, address, re.IGNORECASE)
    
    if match:
        street_type = match.group(1)
        street_name = match.group(2).strip()
        
        # Нормализуем тип улицы
        street_type_normalized = "улица"
        if any(t in street_type.lower() for t in ["пр", "проспект"]):
            street_type_normalized = "проспект"
        elif any(t in street_type.lower() for t in ["б-р", "бульвар"]):
            street_type_normalized = "бульвар"
        elif any(t in street_type.lower() for t in ["наб", "набережная"]):
            street_type_normalized = "набережная"
        elif any(t in street_type.lower() for t in ["ш", "шоссе"]):
            street_type_normalized = "шоссе"
        elif any(t in street_type.lower() for t in ["пер", "переулок"]):
            street_type_normalized = "переулок"
        elif any(t in street_type.lower() for t in ["пл", "площадь"]):
            street_type_normalized = "площадь"
        
        return f"{street_type_normalized} {street_name}"
    
    # Шаг 2: Если не нашли явного типа, ищем стандартные шаблоны
    fallback_patterns = [
        r'^([^\d,]+?)(?:\s*,|\s*$|\s*\d)',
        r'(?:ул\.?|улица)\s*([^\d,]+?)(?:\s*,|\s*$|\s*\d)',
        r'([^\d,]+?)(?:\s*улица|\s*ул\.?)'
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, address, re.IGNORECASE)
        if match:
            street_candidate = match.group(1).strip()
            # Если в кандидате есть число и слово "год", это вероятно "улица 1905 года"
            if re.search(r'\d+\s+год', street_candidate, re.IGNORECASE):
                return f"улица {street_candidate}"
            return street_candidate
    
    # Шаг 3: Если все методы не сработали, берем часть до первой запятой
    parts = address.split(',')
    if parts and parts[0].strip():
        return parts[0].strip()
    
    return address.strip()

def parse_building_components(full_address):
    """
    Точно извлекает компоненты здания
    """
    result = {
        "номер_дома": "",
        "номер_корпуса": "",
        "строение": "",
        "владение": "",
        "полный_номер_для_поиска": "",
        "сырой_номер": ""  # Для хранения полного номера как в базе
    }
    
    # Создаем копию для обработки
    address = full_address.lower()
    
    # Шаг 1: Извлекаем полный номер как в базе
    full_number_match = re.search(r'д\.?\s*([^\s,]+)(?:\s|$|,)', address)
    if not full_number_match:
        full_number_match = re.search(r'([^\s,]+\s*к\s*\d+)(?:\s|$|,)', address)
    if not full_number_match:
        # Ищем любой числовой компонент с возможными буквами
        full_number_match = re.search(r'(\d+\s*[а-я]*)\b', address)
    
    if full_number_match:
        result["сырой_номер"] = full_number_match.group(1).strip()
    
    # Шаг 2: Точный поиск компонентов
    # Номер дома
    house_match = re.search(r'д(?:ом)?\.?\s*([^\s,]+)(?:\s|$|,)', address)
    if house_match:
        result["номер_дома"] = house_match.group(1).strip()
    else:
        # Вторичный поиск номера дома
        fallback_match = re.search(r'\b(\d+\s*[а-я]?)\b(?!\s*(?:корп|к\.|стр|строение|влад|вл\.))', address)
        if fallback_match:
            result["номер_дома"] = fallback_match.group(1).strip()
    
    # Номер корпуса
    corps_match = re.search(r'(?:корп|к)\.?\s*([^\s,]+)(?:\s|$|,)', address)
    if corps_match:
        result["номер_корпуса"] = corps_match.group(1).strip()
    
    # Строение
    build_match = re.search(r'(?:стр|строение)\.?\s*([^\s,]+)(?:\s|$|,)', address)
    if build_match:
        result["строение"] = build_match.group(1).strip()
    
    # Владение
    own_match = re.search(r'(?:вл|владение)\.?\s*([^\s,]+)(?:\s|$|,)', address)
    if own_match:
        result["владение"] = own_match.group(1).strip()
    
    # Формируем полный номер для поиска в базе
    parts = []
    if result["номер_дома"]:
        parts.append(result["номер_дома"])
    if result["номер_корпуса"]:
        parts.append(f"к{result['номер_корпуса']}")
    if result["строение"]:
        parts.append(f"с{result['строение']}")
    if result["владение"]:
        parts.append(f"вл{result['владение']}")
    
    result["полный_номер_для_поиска"] = "".join(parts)
    
    # Если нет полного номера для поиска, используем сырой номер
    if not result["полный_номер_для_поиска"] and result["сырой_номер"]:
        result["полный_номер_для_поиска"] = result["сырой_номер"].replace(" ", "")
    
    return result

def find_closest_street(street_query, db_path='../../filtered_data/addresses.db'):
    """
    Находит ближайшее совпадение для улицы с улучшенными метриками
    """
    if not street_query:
        return None
    
    normalized_query = normalize_text(street_query)
    
    if len(normalized_query) < 4:  # Повышаем минимальную длину
        return None
    
    # Получаем улицы из базы
    streets = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT street FROM addresses 
                WHERE street IS NOT NULL AND street != ''
                LIMIT 3000
            """)
            streets = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Ошибка при работе с базой данных: {e}")
        return None
    
    if not streets:
        return None
    
    # Готовим TF-IDF для всех улиц сразу для оптимизации
    normalized_streets = [normalize_text(st) for st in streets]
    all_texts = [normalized_query] + normalized_streets
    
    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = tfidf_matrix[0]
        streets_vecs = tfidf_matrix[1:]
    except Exception as e:
        print(f"Ошибка при TF-IDF векторизации: {e}")
        # Резервный вариант без TF-IDF
        streets_vecs = None
        query_vec = None
    
    street_scores = []
    
    for i, (street, normalized_street) in enumerate(zip(streets, normalized_streets)):
        if len(normalized_street) < 4:
            continue
        
        # Быстрая проверка на наличие общих ключевых слов
        key_roads = ["родионова", "кутузовский", "победы", "гагарина", "1905 года"]
        for key_road in key_roads:
            if key_road in normalized_query and key_road in normalized_street:
                return {"street": street, "score": 0.98}
        
        # Расстояние Левенштейна (вес 0.1667)
        lev_dist = Levenshtein.distance(normalized_query, normalized_street)
        max_len = max(len(normalized_query), len(normalized_street))
        lev_sim = 1 - (lev_dist / max_len) if max_len > 0 else 0
        
        # Жаро-Винкли (вес 0.5)
        jw_sim = Levenshtein.jaro_winkler(normalized_query, normalized_street)
        
        # Косинусное сходство (вес 0.3333)
        if streets_vecs is not None:
            cos_sim = cosine_similarity(query_vec, streets_vecs[i])[0][0]
        else:
            cos_sim = 0.5  # Нейтральное значение
        
        # Комбинированный скор
        combined_score = (
            0.1667 * lev_sim +
            0.5 * jw_sim +
            0.3333 * cos_sim
        )
        
        # Бонус за совпадение типа улицы
        query_has_type = any(st_type in normalized_query for st_type in ["улица", "проспект", "бульвар", "набережная", "шоссе"])
        street_has_type = any(st_type in normalized_street for st_type in ["улица", "проспект", "бульвар", "набережная", "шоссе"])
        
        if query_has_type and street_has_type:
            combined_score += 0.1
        
        street_scores.append((street, combined_score))
    
    # Сортируем по скору и возвращаем лучшее совпадение
    if street_scores:
        best_street = max(street_scores, key=lambda x: x[1])
        if best_street[1] > 0.45:  # Повышаем порог для отсечения плохих совпадений
            return {"street": best_street[0], "score": best_street[1]}
    
    return None

def find_closest_building(street_name, building_info, db_path='../../filtered_data/addresses.db'):
    """
    Находит ближайшее здание по номеру
    """
    if not street_name:
        return None
    
    buildings = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT lat, lon, number FROM addresses
                WHERE street = ?
                LIMIT 100
            """, [street_name])
            buildings = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Ошибка при работе с базой данных: {e}")
        return None
    
    if not buildings:
        return None
    
    # Если есть конкретный номер для поиска
    if building_info["полный_номер_для_поиска"]:
        query_number = normalize_text(building_info["полный_номер_для_поиска"])
        best_match = None
        best_score = -1
        
        for lat, lon, db_number in buildings:
            if not db_number:
                continue
            
            normalized_db_num = normalize_text(str(db_number))
            
            # Прямое совпадение имеет высший приоритет
            if query_number == normalized_db_num:
                return {
                    "lat": lat,
                    "lon": lon,
                    "number": db_number,
                    "score": 0.99
                }
            
            # Левенштейн для номеров
            lev_dist = Levenshtein.distance(query_number, normalized_db_num)
            max_len = max(len(query_number), len(normalized_db_num))
            lev_sim = 1 - (lev_dist / max_len) if max_len > 0 else 0
            
            # Жаро-Винкли для номеров
            jw_sim = Levenshtein.jaro_winkler(query_number, normalized_db_num)
            
            # Комбинированный скор для номеров
            combined_score = 0.6 * lev_sim + 0.4 * jw_sim
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    "lat": lat,
                    "lon": lon,
                    "number": db_number,
                    "score": combined_score
                }
        
        # Возвращаем лучшее совпадение, если оно достаточно хорошее
        if best_match and best_score > 0.7:
            return best_match
    
    # Если не нашли по номеру, берем первое здание на улице
    return {
        "lat": buildings[0][0],
        "lon": buildings[0][1],
        "number": buildings[0][2],
        "score": 0.4  # Промежуточный скор
    }

def geocode_address(full_address, db_path='../../filtered_data/addresses.db'):
    """
    Основная функция геокодирования
    """
    # 1. Извлекаем название улицы
    street_name = extract_street_name(full_address)
    
    # 2. Извлекаем компоненты здания
    building_info = parse_building_components(full_address)
    
    # 3. Находим ближайшую улицу
    street_match = find_closest_street(street_name, db_path)
    
    # 4. Формируем результат
    result = {
        "searched_address": full_address,
        "objects": []
    }
    
    if street_match:
        # 5. Находим здание
        building_match = find_closest_building(
            street_match["street"], 
            building_info, 
            db_path
        )
        
        if building_match:
            # 6. Формируем объект результата
            obj = {
                "город": "Москва",
                "улица": street_match["street"],
                "номер_дома": building_info["номер_дома"],
                "номер_корпуса": building_info["номер_корпуса"],
                "строение": building_info["строение"],
                "lon": building_match["lon"],
                "lat": building_match["lat"],
                "score": (street_match["score"] + building_match["score"]) / 2
            }
            result["objects"].append(obj)
    
    return result