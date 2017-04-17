# -*- coding: utf-8 -*-

# Scrapy settings for crawler project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

import os

BOT_NAME = 'crawler'

SPIDER_MODULES = ['crawler.spiders']
NEWSPIDER_MODULE = 'crawler.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Set LOGGING
LOG_LEVEL = 'INFO'
#LOG_FILE = os.environ.get('LOG_FILE', 'scrapy.log')

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 5

# Configure a delay for requests for the same website (default: 0)
# See http://scrapy.readthedocs.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 2
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

DATABASE_URL = os.environ.get('DATABASE_URL', '12345')
OPENSTREETMAP_BASE_URL = 'http://nominatim.openstreetmap.org/search/'
GOOGLE_MAP_API_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='

# Proxy
PROXY = os.environ.get('PROXY_URL')
API_SCRAPOXY = os.environ.get('PROXY_API')
API_SCRAPOXY_PASSWORD = os.environ.get('PROXY_PASSWORD', '').encode()

# Enable or disable downloader middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
if PROXY:
    DOWNLOADER_MIDDLEWARES = {
        'crawler.middlewares.crawledURLCheck.CrawledURLCheck': 100,
        'scrapoxy.downloadmiddlewares.proxy.ProxyMiddleware': 101,
        'scrapoxy.downloadmiddlewares.wait.WaitMiddleware': 102,
        'scrapoxy.downloadmiddlewares.scale.ScaleMiddleware': 103,
        'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': None,
    }
else:
    DOWNLOADER_MIDDLEWARES = {
        'crawler.middlewares.crawledURLCheck.CrawledURLCheck': 100,
    }

#SPIDER_MIDDLEWARES = {
#    'crawler.middlewares.CrawlerSpiderMiddleware': 543,
#}

# Enable or disable extensions
# See http://scrapy.readthedocs.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See http://scrapy.readthedocs.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'crawler.pipelines.municipalityFinder.MunicipalityFinderPipeline': 110,
    'crawler.pipelines.objectTypeFinder.ObjectTypeFinderPipeline': 120,
    'crawler.pipelines.duplicateCheck.DuplicateCheckPipeline': 150,
    'crawler.pipelines.tags.TagsPipeline': 160,
    'crawler.pipelines.databaseWriter.DatabaseWriterPipeline': 200,
    #'crawler.pipelines.jsonWriter.JSONWriterPipeline': 300,
}

# custom setting for item pipeline in respider.py
ITEM_PIPELINES_RESPIDER = {
    'crawler.pipelines.tags.TagsPipeline': 160,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See http://doc.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
