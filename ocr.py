import os
import cv2
import re
import math
import tempfile
import json
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import PyPDF4 as pdf
import pytesseract as tess

from text_to_num import alpha2digit as a2d
# from deskew import determine_skew
from typing import Union, Tuple

os.environ['w2n.lang'] = 'es'

# tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class OCRRun:
    def __init__(self) -> None:
        self.tess = tess
        self.ocr_running = True

    def crop_white_border(self, image: cv2.Mat, padding: Union[int, Tuple[int, int]] = 30, center_line_perct: int = 10) -> cv2.Mat:
        '''
        Crop white border of image
        :param image: image to crop
        :param padding: padding to add to the cropped image
        :param center_line_perct: percentage of the image to consider as center line (vertical and horizontal) to search for the white border to crop
        :return: cropped image
        '''

        img = image.copy()
        pad_x = padding if isinstance(padding, int) else padding[0]
        pad_y = padding if isinstance(padding, int) else padding[1]
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        bit = cv2.bitwise_not(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))

        vsigma = bit.shape[1]*center_line_perct//100
        vcenter = bit.shape[1]//2
        center_vline = bit[:, vcenter-vsigma:vcenter+vsigma]
        center_vline = np.sum(center_vline, axis=1)
        min_vindex = np.where(center_vline > 255*5)[0][0]
        max_vindex = np.where(center_vline > 255*5)[0][-1]

        # min_vindices = []
        # max_vindices = []

        # for i in range(bit.shape[0] - vsigma):
        #     vline = bit[i:i+vsigma, :]
        #     vline_sum = np.sum(vline)
        #     if vline_sum > 255*5:
        #         min_vindices.append(i)
        #         max_vindices.append(i+vsigma)

        # min_vindex = int(np.mean(min_vindices))
        # max_vindex = int(np.mean(max_vindices))

        if min_vindex-(pad_x+1) > 0:
            min_vindex = min_vindex - pad_x
        if max_vindex+(pad_x+1) < bit.shape[0]:
            max_vindex = max_vindex+pad_x

        img = img[min_vindex:max_vindex, :, :]

        blur = cv2.GaussianBlur(img, (3, 3), 0)
        bit = cv2.bitwise_not(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))
        hsigma = bit.shape[0]*center_line_perct//100
        hcenter = bit.shape[0] // 2
        center_hline = bit[hcenter-hsigma:hcenter+hsigma, :]
        center_hline = np.sum(center_hline, axis=0)
        min_hindex = np.where(center_hline > 255*5)[0][0]
        max_hindex = np.where(center_hline > 255*5)[0][-1]
        if min_hindex-(pad_y+1) > 0:
            min_hindex = min_hindex-pad_y
        if max_hindex+(pad_y+1) < bit.shape[1]:
            max_hindex = max_hindex+pad_y
        img = img[:, min_hindex:max_hindex, :]
        return img

    def rotate(self, image: cv2.Mat, angle: float, background: Union[int, Tuple[int, int, int]]) -> cv2.Mat:
        '''
        Rotate an image by angle degrees.
        :param image: image to be rotated
        :param angle: angle in degrees
        :param background: background color
        '''

        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian)*old_height) +\
            abs(np.cos(angle_radian)*old_width)
        height = abs(np.sin(angle_radian)*old_width) +\
            abs(np.cos(angle_radian)*old_height)

        image_center = tuple(np.array(image.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width)/2
        rot_mat[0, 2] += (height - old_height)/2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    def erase_non_black_text(self, img: cv2.Mat, lower: int = 0, upper: int = 150) -> cv2.Mat:
        '''	
        Erase non black text from an image.
        :param img: image to be processed
        :param lower: lower threshold
        :param upper: upper threshold
        '''
        lower = np.array([lower, lower, lower])
        upper = np.array([upper, upper, upper])
        image = cv2.inRange(img, lower, upper)
        image = cv2.bitwise_not(image)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[image == 0] = 255
        image[mask == 0] = 255
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def enhance_image(self, img: cv2.Mat, gamma: int = 2) -> cv2.Mat:
        ''' 
        Enhance an image.
        :param img: image to be enhanced
        :param gamma: gamma value
        '''
        image = cv2.pow(img/255., gamma)
        image = (image*255).astype(np.uint8)
        return image

    def run_ocr(self, pdf_folder: str, extras_folder: str, create_pdf_folder: bool, create_text_folder: bool, create_processed_images_folder: bool, progress_callback: tuple) -> None:
        # print("Iniciando OCR...")
        self.ocr_running = True
        pdf_folder = pdf_folder
        extras_folder = extras_folder
        # progress_callback(0, "Iniciando OCR...")

        create_pdf_folder = create_pdf_folder
        if create_pdf_folder:
            ruta_buscables = os.path.join(extras_folder, 'contratos_buscables')
            if not os.path.exists(ruta_buscables):
                os.makedirs(ruta_buscables)

        create_text_folder = create_text_folder
        if create_text_folder:
            ruta_textos = os.path.join(extras_folder, 'textos')
            if not os.path.exists(ruta_textos):
                os.makedirs(ruta_textos)

        create_processed_images_folder = create_processed_images_folder
        if create_processed_images_folder:
            ruta_imagenes = os.path.join(extras_folder, 'imagenes_procesadas')
            if not os.path.exists(ruta_imagenes):
                os.makedirs(ruta_imagenes)

        config = '--psm 1'

        total_pages = 0
        cumulative_progress = 0
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                total_pages += pdf.PdfFileReader(
                    open(pdf_folder+os.sep+file, 'rb')).getNumPages()

        text_dict = {'text': [], 'page': [], 'file': []}

        for file in os.listdir(pdf_folder):
            if not self.ocr_running:
                break
            if file.endswith('.pdf'):
                merged_pdf = pdf.PdfFileMerger()
                text = ''
                pdf_reads = pdfium.PdfDocument(
                    open(pdf_folder+os.sep+file, 'rb'))
                for i, page in enumerate(pdf_reads):
                    if not self.ocr_running:
                        break
                    image = page.render(scale=3).to_pil()
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # angle = determine_skew(gray)
                    # if angle is None:
                    #     angle = 0
                    # image = self.rotate(image, angle, (255, 255, 255))
                    # image = crop_white_border(image)
                    image = self.erase_non_black_text(image)
                    if create_processed_images_folder:
                        cv2.imwrite(ruta_imagenes+os.sep +
                                    file[:-4]+'_'+str(i)+'.png', image)

                    if create_text_folder:
                        page_text = ''
                        page_text = tess.image_to_string(
                            image, config=config, lang='spa')
                        page_text = page_text.split('\n')
                        page_text = ' '.join(page_text)
                        page_text = a2d(page_text.replace(
                            'guión', '-'), lang='es', relaxed=True)
                        text += page_text

                        text_dict['text'].append(page_text)
                        text_dict['page'].append(i+1)
                        text_dict['file'].append(file)

                    if create_pdf_folder:
                        smaller_image = cv2.resize(
                            image, (0, 0), fx=.5, fy=.5)

                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_page_path = os.path.join(
                                temp_dir, 'temp_page.pdf')
                            with open(temp_page_path, 'wb') as f:
                                f.write(tess.image_to_pdf_or_hocr(
                                    smaller_image, extension='pdf', lang='spa', config=config))

                            with open(temp_page_path, 'rb') as f:
                                temp_pdf = pdf.PdfFileReader(f)
                                merged_pdf.append(
                                    temp_pdf, import_bookmarks=False)

                    cumulative_progress += 1
                    progress = (cumulative_progress/total_pages)*100
                    nombre_archivo = file.replace('.pdf', '')
                    # n_letras = 10
                    # nombre_archivo = nombre_archivo[-n_letras:] if len(
                    #     nombre_archivo) > n_letras else nombre_archivo
                    message = f'Archivo: {nombre_archivo} \n Progreso: {progress:.2f}% | Página {i+1} de {len(pdf_reads)}'
                    progress_callback(progress, message)

                if create_pdf_folder:
                    with open(ruta_buscables+os.sep+file[:-4]+'.pdf', 'wb') as f:
                        merged_pdf.write(f)

                if create_text_folder:
                    with open(ruta_textos+os.sep+file[:-4]+'.txt', 'w', encoding='utf-8') as f:
                        f.write(text)
                    with open(ruta_textos+os.sep+file[:-4]+'.json', 'w', encoding='utf-8') as f:
                        json.dump(text_dict, f)

    def create_excel_file(self, text_folder: str) -> None:
        textos = pd.DataFrame(columns=['text', 'page', 'file'])
        ruta_textos = os.path.join(text_folder, 'textos')

        for file in os.listdir(ruta_textos):
            if file.endswith('.json'):
                textos = pd.concat([textos, pd.read_json(
                    ruta_textos+os.sep+file)], ignore_index=True)

        textos.drop_duplicates(inplace=True)

        extracted_data = pd.DataFrame()

        for _, cols in textos.iterrows():
            results = re.findall(
                r'(\d+(\.\d+)?)(\s)?(m²|m2|metros cuadrados|metro cuadrado|metro)', cols['text'])
            if len(results) > 0:
                for result in results:
                    extracted_data = pd.concat([extracted_data, pd.DataFrame(
                        {'file': cols['file'], 'page': cols['page'], 'area': float(result[0])}, index=[0])], ignore_index=True)
        extracted_data.to_excel(
            ruta_textos+os.sep+'resultados.xlsx', index=False)
        # print('Archivo creado en: '+ruta_textos+os.sep+'resultados.xlsx')

    def stop_ocr(self):
        self.ocr_running = False


if __name__ == '__main__':
    ocr = OCRRun()
    ocr.run_ocr(
        pdf_folder='/Users/jonatan/Desktop/pdfs',
        create_pdf_folder=True,
        create_text_folder=True,
        create_processed_images_folder=True,
        extras_folder='/Users/jonatan/Desktop/pdfs/extras',
        progress_callback=lambda progress, message: print(message)
    )
    #ocr.create_excel_file('/Users/jonatan/Desktop/pdfs/extras')
