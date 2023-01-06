import logging
import os
import pickle
import sys
from abc import abstractmethod

from mega import Mega
from zipfile_deflate64 import ZipFile
import wget
from src import manifest
from src.data.data_container import DataContainer
import validators
import urllib.parse

logger = logging.getLogger('data_provider')


class DataProvider:
    @abstractmethod
    def collect(self, args) -> DataContainer:
        pass


class PickleDataProvider(DataProvider):
    def __init__(self, file_path):
        self.uri = file_path

    def collect(self, **args) -> DataContainer:
        self._handle_url(**args)
        file = open(self.uri, 'rb')
        return pickle.load(file)

    @staticmethod
    def save(container, file_path):
        file = open(file_path, 'wb')
        pickle.dump(container, file)

    def _handle_url(self, **kwargs):
        if not validators.url(self.uri):
            return

        file_name = kwargs['file_name'] if 'file_name' in kwargs else None
        if file_name is None:
            raise Exception('missing required parameter [file_name]')

        local_file = manifest.DATA_PATH + file_name
        if self._file_exists(self.as_pkl(local_file)):
            logger.info(f'file exists locally, loading path {local_file}')
        else:
            self._get(self.uri, self.as_zip(local_file))
        self.uri = self.as_pkl(local_file)

    def _bar_progress(self, current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    def _file_exists(self, downloaded_file):
        return os.path.isfile(downloaded_file)

    def _get(self, url, into):
        try:
            logger.info(f'downloading file into {into}')
            self._download(url, into)
            logger.info('extracting...')
            with ZipFile(into, 'r') as zipObj:
                zipObj.extractall(manifest.DATA_PATH)
            logger.info('loading...')
            return True
        except Exception as e:
            logger.info(f'error while downloading the file {e}')
            raise e
        finally:
            if self._file_exists(into):
                os.remove(into)

    def _download(self, url, full_path):
        if 'mega.nz' in url:
            logger.info('mega.nz detected, using mega downloader...')
            fname = full_path.split('/')[-1]
            directory = "/".join(full_path.split('/')[0:-1]) + "/"
            self._mega_downloader(url, directory)
        else:
            self._wget_downloader(url, full_path)

    def _wget_downloader(self, url, into):
        wget.download(url, into, bar=self._bar_progress)

    def _mega_downloader(self, url, directory):
        try:
            m = Mega().login()
            m.download_url(url, directory)
        except Exception as ignore:
            pass
        return True

    def as_zip(self, a_file):
        return a_file + '.zip'

    def as_pkl(self, a_file):
        return a_file + '.pkl'
