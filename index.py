import requests
import json
import logging
from typing import List, Dict, Any
from transformers import pipeline
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Заголовки для запросов
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9"
}

# Ключевые слова для фильтрации
keywords_vish = ["инженерная школа", "РУТ МИИТ", "ВИШ"]
keywords_high_speed = ["ВСМ", "скоростные магистрали", "высокоскоростной"]
keywords_rzd = ["РЖД", "Российские железные дороги"]

class NewsFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def fetch_news(self, query: str) -> List[Dict[str, Any]]:
        url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&apiKey={self.api_key}"
        logging.info(f"Запрос данных NewsAPI по запросу: {query} с URL: {url}")
        try:
            response = self.session.get(url, headers=HEADERS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при запросе NewsAPI: {e}")
            if response is not None:
                logging.error(f"HTTP Status Code: {response.status_code}")
                logging.error(f"Response Content: {response.text}")
            return []

        articles = response.json().get('articles', [])
        if not articles:
            logging.info(f"Для {query} не найдено статей в ответе NewsAPI.")
        
        fetched_articles = [{"title": article['title'], "link": article['url'], "published": article['publishedAt']} for article in articles]
        logging.debug(f"Извлеченные статьи: {fetched_articles}")
        return fetched_articles

    def fetch_vish_news(self) -> List[Dict[str, Any]]:
        return self.fetch_news('Высшая инженерная школа')

    def fetch_high_speed_railways(self) -> List[Dict[str, Any]]:
        return self.fetch_news('ВСМ')

    def fetch_russian_railways(self) -> List[Dict[str, Any]]:
        return self.fetch_news('РЖД Российские Железные дороги')

class NewsParser:
    def __init__(self, fetcher: NewsFetcher):
        self.fetcher = fetcher
        self.sentiment_model = pipeline('sentiment-analysis', model="blanchefort/rubert-base-cased-sentiment")

    def filter_and_sort_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.info("Сортировка статей по дате публикации.")
        return sorted(articles, key=lambda x: x['published'], reverse=True)

    def analyze_sentiment(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.info("Анализ настроений для статей.")
        for article in articles:
            title = article['title']
            result = self.sentiment_model(title)[0]
            article['sentiment'] = result['label']
            article['subjectivity'] = result['score']
            logging.debug(f"Заголовок: {article['title']}, Настроение: {article['sentiment']}, Субъективность: {article['subjectivity']}")
        return articles

    def adjust_subjectivity(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.info("Корректировка субъективности для нейтральных статей.")
        for article in articles:
            if article['sentiment'] == 'NEUTRAL':
                article['subjectivity'] = 0.1  # Устанавливаем фиксированное значение для нейтральных статей
        return articles

    def create_dashboard(self, data: List[Dict[str, Any]]) -> str:
        logging.info("Создание HTML для панели управления.")
        dashboard = """
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Новостная панель</title>
            <style>
                body { font-family: Arial, sans-serif; }
                h1 { color: #333; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 10px 0; }
                a { text-decoration: none; color: #1a0dab; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Новостная панель</h1>
            <ul>
        """
        for entry in data:
            sentiment_color = "green" if entry['sentiment'] == 'POSITIVE' else "red" if entry['sentiment'] == 'NEGATIVE' else "gray"
            dashboard += f"<li><a href='{entry['link']}' style='color:{sentiment_color}'>{entry['title']}</a> - {entry['published']} (Настроение: {entry['sentiment']}, Субъективность: {entry['subjectivity']:.2f})</li>"
        dashboard += """
            </ul>
        </body>
        </html>
        """
        return dashboard

    def save_dashboard(self, html_content: str, filename: str) -> None:
        logging.info(f"Сохранение панели в файл {filename}")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def save_to_json(self, result: List[Dict[str, Any]], filename: str) -> None:
        logging.info(f"Сохранение результатов в JSON файл: {filename}")
        with open(filename, "w", encoding="utf-8") as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        logging.info(json.dumps(result, ensure_ascii=False, indent=4))

    def read_json(self) -> List[Dict[str, Any]]:
        logging.info("Чтение результатов из JSON.")
        with open("results.json", "r", encoding="utf-8") as outfile:
            data = json.load(outfile)
        return data

    def filter_articles_by_keywords(self, articles: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        logging.info("Фильтрация статей по ключевым словам.")
        filtered_articles = []
        for article in articles:
            title = article.get('title', '')
            summary = article.get('description', '')  
            if any(keyword in title or keyword in summary for keyword in keywords):
                filtered_articles.append(article)
        return filtered_articles

    def main(self):
        logging.info("Извлечение новостных статей.")
        vish_news = self.fetcher.fetch_vish_news()
        high_speed_news = self.fetcher.fetch_high_speed_railways()
        russian_railways_news = self.fetcher.fetch_russian_railways()

        all_articles = vish_news + high_speed_news + russian_railways_news

        logging.info("Удаление дублирующих статей.")
        unique_articles = {entry['link']: entry for entry in all_articles}.values()

        logging.info("Сортировка и анализ статей.")
        sorted_articles = self.filter_and_sort_articles(list(unique_articles))
        analyzed_articles = self.analyze_sentiment(sorted_articles)
        adjusted_articles = self.adjust_subjectivity(analyzed_articles)

        logging.info("Фильтрация статей по ключевым словам.")
        filtered_vish = self.filter_articles_by_keywords(adjusted_articles, keywords_vish)
        filtered_high_speed = self.filter_articles_by_keywords(adjusted_articles, keywords_high_speed)
        filtered_rzd = self.filter_articles_by_keywords(adjusted_articles, keywords_rzd)

        final_articles = filtered_vish + filtered_high_speed + filtered_rzd

        # Объединение всех данных в один JSON-файл
        result = {
            "unique_articles": list(unique_articles),
            "analyzed_articles": analyzed_articles,
            "filtered_vish": filtered_vish,
            "filtered_high_speed": filtered_high_speed,
            "filtered_rzd": filtered_rzd,
            "final_articles": final_articles
        }

        logging.info("Сохранение всех данных в JSON файл.")
        self.save_to_json(result, 'all_articles.json')

        logging.info("Создание и сохранение панели.")
        dashboard_content = self.create_dashboard(final_articles)
        self.save_dashboard(dashboard_content, 'news_dashboard.html')

# Запуск основной функции
if __name__ == "__main__":
    api_key = '87db1ec85f5645df82e8a9d425a2b911'
    fetcher = NewsFetcher(api_key)
    parser = NewsParser(fetcher)
    parser.main()
