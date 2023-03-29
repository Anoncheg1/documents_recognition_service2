import fitz
import PIL.Image
import io
import os
from pdf2image import convert_from_path  # poppler-utils
#own
from logger import logger as log

DPI = 300

# 19.603 s  # PyMuPDF + poppler
def pdf2png(pdf_filename: str, tmpdir: str) -> list:
    """

    :param pdf_filename:
    :param tmpdir: to extract to
    :return: tmpdir + file_name
    """
    returnlist = []

    pdf_file = fitz.open(pdf_filename)
    # check that all pages have 1 image
    multi_image_files = [len(pdf_file[i].getImageList()) != 1 for i in range(len(pdf_file))]
    if any(multi_image_files):
        log.debug("pdf_converter: More than one image at one page.")
        return pdf2png_poppler(pdf_filename, tmpdir)

    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.getImageList()

        # -- just first
        # get the XREF of the image
        xref = image_list[0][0]
        # extract the image bytes
        base_image = pdf_file.extractImage(xref)
        image_bytes = base_image["image"]

        # load it to PIL
        image = PIL.Image.open(io.BytesIO(image_bytes))
        # replace with slow version if rare bug
        w, h = image.size
        if w < 400 or h < 400:
            return pdf2png_poppler(pdf_filename, tmpdir)
        # save it to local disk
        filename = os.path.join(tmpdir, 'PNG' + str(page_index) + '.png')
        image.save(open(filename, "wb"))
        returnlist.append(filename)
        log.debug('File successfully extracted from PDF, count:' + str(len(returnlist)))

    return returnlist


# 86.615 s
def pdf2png_poppler(pdf_filename: str, tmpdir: str, thread_count=1) -> list:
    """
    Конвертация PDF -> PNG

    :param pdf_filename: path to pdf
    :param tmpdir: ends without /
    :param thread_count:
    :return: Возвращает list с именами сохраненных файлов PNG в этой же директории
    """

    # pdf2image.exceptions.PDFPopplerTimeoutError: Run poppler poppler timeout.
    # is catched at MainOpenCV
    convert_from_path(pdf_filename, dpi=DPI, use_pdftocairo=True, thread_count=thread_count,
                      output_folder=tmpdir, timeout=60)
    files = os.listdir(tmpdir)
    if len(files) >= 100:
        raise ValueError("Too many files after pdf2image")
    # filter out pdf, sort by xxx-01.png
    files = [(int(f[-6:][:2]), os.path.join(tmpdir, f)) for f in files if f[-4:].lower() == '.png']
    files = sorted(files, key=lambda x: x[1])
    _, returnlist = zip(*files)

    log.debug('Total {} file successfully converted and saved from PDF '.format(str(len(returnlist))))
    return returnlist

#
# # from wand.image import Image, Color  # imagemagic
# # -- slow --
# def pdf2png_wand(pdf_filename: str, tmpdir: str) -> list:
#     returnlist = []
#     pages = Image(filename=pdf_filename, resolution=DPI)  # PDF will have several pages.
#     for ii, img in enumerate(pages.sequence):
#         filename = tmpdir + '/PNG' + str(ii) + '.png'
#         with Image(img) as i:
#             i.format = 'png'
#             i.background_color = Color('white')  # Set white background.
#             i.alpha_channel = 'remove'  # Remove transparency and replace with bg.
#             i.save(filename=filename)  # and create directory
#         returnlist.append(filename)
#     return returnlist

# if __name__ == '__main__':  # test
#     from utils.profiling import profiling_before, profiling_after
#     pr = profiling_before()
#     p = '/home/u2/Downloads/документы.pdf'
#     pdf2png_poppler(p, tmpdir= './1')
#     pdf2png_poppler(p, tmpdir='./1')
#     pdf2png_poppler(p, tmpdir='./1')
#     profiling_after(pr)  # noqa