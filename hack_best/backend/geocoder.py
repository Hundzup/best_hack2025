import pandas as pd
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import rapidfuzz
from rapidfuzz import fuzz

class MoscowGeocoder:
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация геокодера для Москвы
        
        Parameters:
        df (pd.DataFrame): DataFrame со списком объектов и их координатами
        """
        # Исправление перепутанных координат
        self.data = df.copy()
        self.data['lon'] = df['y']  # долгота
        self.data['lat'] = df['x']  # широта
        
        # Загрузка классификатора сокращений
        self.abbreviations = self._load_abbreviations()
        
        # Создание TF-IDF векторизатора для косинусного сходства
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        
        # Нормализация названий и подготовка данных
        self._preprocess_data()
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """
        Загрузка классификатора сокращений адресов согласно Приказу Минфина России от 5 ноября 2015 г. N 171н
        """
        abbreviations = {}
        
        # Субъекты РФ
        abbreviations.update({
            "респ": "республика",
            "респ.": "республика",
            "край": "край",
            "обл": "область",
            "обл.": "область",
            "г.ф.з": "город федерального значения",
            "г.ф.з.": "город федерального значения",
            "а.обл": "автономная область",
            "а.обл.": "автономная область",
            "а.окр": "автономный округ",
            "а.окр.": "автономный округ"
        })
        
        # Муниципальные образования
        abbreviations.update({
            "м.р-н": "муниципальный район",
            "г.о.": "городской округ",
            "г.п.": "городское поселение",
            "с.п.": "сельское поселение",
            "вн.р-н": "внутригородской район",
            "вн.тер.г.": "внутригородская территория города федерального значения"
        })
        
        # Административно-территориальные единицы и населенные пункты
        abbreviations.update({
            "пос": "поселение",
            "пос.": "поселение",
            "р-н": "район",
            "с/с": "сельсовет",
            "г": "город",
            "г.": "город",
            "пгт": "поселок городского типа",
            "пгт.": "поселок городского типа",
            "рп": "рабочий поселок",
            "рп.": "рабочий поселок",
            "кп": "курортный поселок",
            "кп.": "курортный поселок",
            "гп": "городской поселок",
            "гп.": "городской поселок",
            "п": "поселок",
            "п.": "поселок",
            "д": "деревня",
            "д.": "деревня",
            "с": "село",
            "с.": "село",
            "сл": "слобода",
            "сл.": "слобода",
            "ст": "станция",
            "ст.": "станция",
            "ст-ца": "станица",
            "у": "улус",
            "у.": "улус",
            "х": "хутор",
            "х.": "хутор",
            "рзд": "разъезд",
            "рзд.": "разъезд",
            "зим": "зимовье",
            "зим.": "зимовье",
            "м-ко": "местечко",
            "м-ко.": "местечко"
        })
        
        # Элементы улично-дорожной сети
        abbreviations.update({
            "ул": "улица",
            "ул.": "улица",
            "ал": "аллея",
            "ал.": "аллея",
            "б-р": "бульвар",
            "наб": "набережная",
            "наб.": "набережная",
            "пер": "переулок",
            "пер.": "переулок",
            "пл": "площадь",
            "пл.": "площадь",
            "пр-д": "проезд",
            "пр-кт": "проспект",
            "пр-кт.": "проспект",
            "туп": "тупик",
            "туп.": "тупик",
            "ш": "шоссе",
            "ш.": "шоссе"
        })
        
        # Идентификационные элементы
        abbreviations.update({
            "д": "дом",
            "д.": "дом",
            "корп": "корпус",
            "корп.": "корпус",
            "стр": "строение",
            "стр.": "строение",
            "влд": "владение",
            "влд.": "владение",
            "зд": "здание",
            "зд.": "здание",
            "кв": "квартира",
            "кв.": "квартира",
            "офис": "офис"
        })
        
        return abbreviations
    
    def _preprocess_data(self):
        """Предварительная обработка данных"""
        # Нормализация названий
        self.data['normalized_name'] = self.data['Name'].apply(self.normalize_address)
        
        # Векторизация для косинусного сходства
        self._vectorize_names()
        
        # Предварительная классификация объектов
        self._classify_objects()
    
    def _classify_objects(self):
        """Классификация объектов для улучшенного распознавания компонентов адреса"""
        # Список известных районов Москвы
        moscow_districts = [
            'арбат', 'басманный', 'волково', 'головинский', 'даниловский', 
            'замоскворечье', 'зюзино', 'ивановское', 'измайлово', 'коньково', 
            'кунцево', 'левобережный', 'митино', 'нагатино-садовники', 
            'останкинский', 'преображенское', 'район', 'савелки', 'соколиная гора', 
            'теплый стан', 'филевский парк', 'хамовники', 'черемушки', 
            'чертаново', 'щукино'
        ]
        
        # Список типов объектов (станции метро, ж/д станции и т.д.)
        object_types = {
            'metro_station': ['станция метро', 'метро'],
            'railway_station': ['станция', 'ж/д ст'],
            'platform': ['платформа'],
            'area': ['район', 'квартал', 'микрорайон']
        }
        
        # Добавление колонок для классификации
        self.data['is_moscow_district'] = False
        self.data['object_type'] = 'other'
        
        for idx, row in self.data.iterrows():
            name_lower = row['Name'].lower()
            
            # Проверка на район Москвы
            for district in moscow_districts:
                if district in name_lower:
                    self.data.at[idx, 'is_moscow_district'] = True
                    break
            
            # Проверка на тип объекта
            found_type = False
            for obj_type, keywords in object_types.items():
                for keyword in keywords:
                    if not found_type and keyword in name_lower:
                        self.data.at[idx, 'object_type'] = obj_type
                        found_type = True
                        break
    
    def _vectorize_names(self):
        """Создание TF-IDF векторов для всех названий"""
        normalized_names = self.data['normalized_name'].fillna('').tolist()
        normalized_names = [name if name.strip() else ' ' for name in normalized_names]
        try:
            self.name_vectors = self.vectorizer.fit_transform(normalized_names)
        except Exception as e:
            print(f"Ошибка при векторизации: {e}")
            self.name_vectors = None
    
    def normalize_address(self, text: str) -> str:
        """
        Нормализация адреса согласно классификатору сокращений
        
        Parameters:
        text (str): Исходный текст адреса
        
        Returns:
        str: Нормализованный адрес
        """
        if not isinstance(text, str) or pd.isna(text) or not text.strip():
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление лишних пробелов и специальных символов, кроме дефисов и точек
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Замена сокращений на полные формы согласно классификатору
        for abbr, full in self.abbreviations.items():
            pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
            text = re.sub(pattern, full, text)
        
        # Дополнительная нормализация: замена множественных пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _jaro_winkler_similarity(self, str1: str, str2: str) -> float:
        """Вычисление схожести по метрике Jaro-Winkler"""
        if not str1 or not str2:
            return 0.0
        return fuzz.WRatio(str1, str2) / 100.0
    
    def _cosine_similarity(self, str1: str, str2: str) -> float:
        """Вычисление косинусного сходства между двумя строками"""
        if not str1 or not str2:
            return 0.0
        
        # Векторизация двух строк
        combined = [str1, str2]
        try:
            vectors = self.vectorizer.transform(combined)
            cos_dist = cosine(vectors[0].toarray()[0], vectors[1].toarray()[0])
            return 1 - cos_dist
        except Exception as e:
            return 0.0
    
    def parse_full_address(self, address: str) -> Dict[str, str]:
        """
        Парсинг полного адреса с извлечением всех компонентов
        
        Parameters:
        address (str): Полный адрес для разбора
        
        Returns:
        Dict[str, str]: Словарь с компонентами адреса
        """
        if not isinstance(address, str) or not address.strip():
            return {
                "город": "",
                "улица": "",
                "номер_дома": "",
                "номер_корпуса": "",
                "строение": "",
                "название": ""
            }
        
        # Нормализация адреса
        norm_address = self.normalize_address(address)
        original_address = address.strip()
        
        # Инициализация результатов
        result = {
            "город": "",
            "улица": "",
            "номер_дома": "",
            "номер_корпуса": "",
            "строение": "",
            "название": original_address  # По умолчанию все название
        }
        
        # 1. Определение города
        city_patterns = [
            r'город\s+([\w\s\-]+?)(?:\s+(?:район|улица|дом|корпус|строение|\d|$))',
            r'г\.([\w\s\-]+?)(?:\s+(?:р-н|ул|д|корп|стр|\d|$))',
            r'г\s+([\w\s\-]+?)(?:\s+(?:р-н|ул|д|корп|стр|\d|$))',
            r'(москва|москвы|москве|москву|москвой)',
            r'(санкт-петербург|санкт-петербурге|питер)',
            r'(новосибирск|екатеринбург|казань|самара|омск|ростов-на-дону|уфа|красноярск|воронеж|пермь)'
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, norm_address, re.IGNORECASE)
            if match:
                city_name = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                result["город"] = city_name.capitalize()
                break
        
        # 2. Определение улицы
        street_patterns = [
            r'улица\s+([\w\s\-]+?)(?:\s+(?:дом|д|корпус|корп|строение|стр|\d|$))',
            r'ул\.([\w\s\-]+?)(?:\s+(?:д|корп|стр|\d|$))',
            r'ул\s+([\w\s\-]+?)(?:\s+(?:д|корп|стр|\d|$))',
            r'проспект\s+([\w\s\-]+?)(?:\s+(?:дом|д|корпус|корп|строение|стр|\d|$))',
            r'пр-кт\s+([\w\s\-]+?)(?:\s+(?:д|корп|стр|\d|$))',
            r'бульвар\s+([\w\s\-]+?)(?:\s+(?:дом|д|корпус|корп|строение|стр|\d|$))',
            r'б-р\s+([\w\s\-]+?)(?:\s+(?:д|корп|стр|\d|$))',
            r'набережная\s+([\w\s\-]+?)(?:\s+(?:дом|д|корпус|корп|строение|стр|\d|$))',
            r'наб\.([\w\s\-]+?)(?:\s+(?:д|корп|стр|\d|$))'
        ]
        
        for pattern in street_patterns:
            match = re.search(pattern, norm_address, re.IGNORECASE)
            if match:
                street_name = match.group(1).strip()
                result["улица"] = street_name
                break
        
        # 3. Определение номера дома
        house_patterns = [
            r'дом\s+(\d+[\wа-я]*)',
            r'д\.\s*(\d+[\wа-я]*)',
            r'д\s+(\d+[\wа-я]*)',
            r'\s+(\d+[\wа-я]*)\s+(?:корпус|корп|строение|стр|$)',
            r'(?:улица|ул|проспект|пр-кт|бульвар|б-р|набережная|наб)\s+[\w\s\-]+?\s+(\d+[\wа-я]*)'
        ]
        
        for pattern in house_patterns:
            match = re.search(pattern, norm_address)
            if match:
                house_number = match.group(1).strip()
                result["номер_дома"] = house_number
                break
        
        # 4. Определение номера корпуса
        building_patterns = [
            r'корпус\s+(\d+[\wа-я]*)',
            r'корп\.\s*(\d+[\wа-я]*)',
            r'к\.\s*(\d+[\wа-я]*)',
            r'корп\s+(\d+[\wа-я]*)',
            r',\s*к\s*(\d+[\wа-я]*)',
            r'\s+к\s*(\d+[\wа-я]*)\b'
        ]
        
        for pattern in building_patterns:
            match = re.search(pattern, norm_address)
            if match:
                building_number = match.group(1).strip()
                result["номер_корпуса"] = building_number
                break
        
        # 5. Определение строения
        structure_patterns = [
            r'строение\s+(\d+[\wа-я]*)',
            r'стр\.\s*(\d+[\wа-я]*)',
            r'с\.\s*(\d+[\wа-я]*)',
            r'стр\s+(\d+[\wа-я]*)'
        ]
        
        for pattern in structure_patterns:
            match = re.search(pattern, norm_address)
            if match:
                structure_number = match.group(1).strip()
                result["строение"] = structure_number
                break
        
        # 6. Если город не найден, пытаемся определить по контексту
        if not result["город"]:
            # Проверяем наличие московских районов в адресе
            moscow_districts = ['центральный', 'северный', 'арбат', 'басманный', 'зюзино', 'коньково', 'кунцево', 'щерибино']
            if any(district in norm_address.lower() for district in moscow_districts):
                result["город"] = "Москва"
            
            # Если в адресе есть "москва" в любом падеже
            if re.search(r'москв[аыуеой]', norm_address, re.IGNORECASE):
                result["город"] = "Москва"
        
        # 7. Если ничего не найдено, используем эвристики для определения названия
        if not result["улица"] and not result["номер_дома"]:
            # Если это похоже на название станции или объекта
            if any(word in norm_address.lower() for word in ['станция', 'метро', 'платформа', 'поселок', 'район']):
                result["название"] = original_address
        
        # 8. Финальная очистка и форматирование
        for key in result:
            if isinstance(result[key], str):
                result[key] = result[key].strip()
        
        return result
    
    def combined_geocoding(self, query: str, threshold: float = 0.6, max_results: int = 1) -> Dict[str, Any]:
        """
        Комбинированный алгоритм геокодирования
        
        Parameters:
        query (str): Поисковый запрос (адрес)
        threshold (float): Порог схожести для отбора результатов (от 0 до 1)
        max_results (int): Максимальное количество возвращаемых результатов
        
        Returns:
        Dict[str, Any]: Результат геокодирования в требуемом формате
        """
        normalized_query = self.normalize_address(query)
        
        results = []
        
        # Поиск совпадений для каждого объекта в датасете
        for idx, row in self.data.iterrows():
            normalized_name = row['normalized_name']
            
            # Пропускаем пустые названия
            if not normalized_name or normalized_name.strip() == '':
                continue
            
            # 1. Расстояние Левенштейна
            lev_dist = rapidfuzz.distance.Levenshtein.distance(normalized_query, normalized_name)
            max_len = max(len(normalized_query), len(normalized_name))
            lev_similarity = 1 - lev_dist / max_len if max_len > 0 else 0
            
            # 2. Jaro-Winkler расстояние
            jw_similarity = self._jaro_winkler_similarity(normalized_query, normalized_name)
            
            # 3. Косинусное сходство
            cos_similarity = self._cosine_similarity(normalized_query, normalized_name)
            
            # Комбинирование метрик с весами
            weights = {
                'levenshtein': 0.1667,
                'jaro_winkler': 0.5,
                'cosine': 0.3333
            }
            
            combined_similarity = (
                weights['levenshtein'] * lev_similarity +
                weights['jaro_winkler'] * jw_similarity +
                weights['cosine'] * cos_similarity
            )
            
            # Финальный score
            final_score = combined_similarity
            
            # Если схожесть выше порога, добавляем в результаты
            if final_score >= threshold:
                # Извлекаем все компоненты адреса
                address_components = self.parse_full_address(row['Name'])
                
                results.append({
                    "город": address_components["город"],
                    "улица": address_components["улица"],
                    "номер_дома": address_components["номер_дома"],
                    "номер_корпуса": address_components["номер_корпуса"],
                    "строение": address_components["строение"],
                    "название": address_components["название"],
                    "lon": row['lon'],
                    "lat": row['lat'],
                    "score": final_score,
                    "metrics": {
                        "levenshtein": lev_similarity,
                        "jaro_winkler": jw_similarity,
                        "cosine": cos_similarity
                    }
                })
        
        # Сортировка по score в порядке убывания
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Ограничение количества результатов
        results = results[:max_results]
        
        return {
            "searched_address": query,
            "objects": results
        }