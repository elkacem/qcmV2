from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from .models import Section, Image
from .forms import SectionForm, ImageForm
import io
from urllib.parse import quote
from django.db.models import Q

import openpyxl

from uuid import uuid4

import os
import cv2
import pandas as pd

import numpy as np

from rich import print

import PIL
import os.path

from django.urls import reverse_lazy
from django.contrib import messages
from django.views.generic.edit import DeleteView


def home(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    section = Section.objects.filter(
        Q(name__icontains=q) |
        Q(description__icontains=q)
    )
    sections = Section.objects.all()[0:5]
    section_count = sections.count()
    context = {'section': section, 'sections': sections, 'section_count': section_count}

    return render(request, 'base/home.html', context)


def sectionsPage(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    sections = Section.objects.filter(name__icontains=q)
    return render(request, 'base/sections.html', {'sections': sections})


def createSection(request):
    form = SectionForm()
    if request.method == 'POST':
        form = SectionForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('home')

    context = {'form': form}

    return render(request, 'base/section_form.html', context)

def deleteSection(request, pk):
    section = Section.objects.get(id=pk)

    if request.method == 'POST':
        section.delete()
        return redirect('home')
    return render(request, 'base/delete.html', {'obj':section})

# def delete_image(request, pk):
#     section = Section.objects.get(id=pk)
#     images = Image.objects.filter(section=section)
#     if request.method == 'POST':
#         images.image.url.delete()
#         return redirect('section-detail', id=pk)
#     context = {
#         'section': section,
#     }
#     return render(request, 'base/delete_image.html', context)


def delete_image(request, image_id):
    section = get_object_or_404(Section, id=25)
    image = get_object_or_404(Image, id=image_id, section=section)

    # Delete the image file from the file system.
    image.image.delete()

    # Delete the image object from the database.
    image.delete()

    # Redirect the user to the section detail page.
    return redirect('section-detail')



def sectionDetail(request):
    section = get_object_or_404(Section, id=25)
    # images = Image.objects.get(pk=pk)
    images = Image.objects.filter(section=section)

    image_count = images.count()
    form = ImageForm()
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            images = request.FILES.getlist('image')
            for image in images:
                Image.objects.create(section=section, image=image)
            return redirect(request.path)
    context = {'section': section, 'form': form, 'images': images, 'image_count': image_count}
    return render(request, 'base/section_detail.html', context)


def download_excel(request):
    print('request', request.GET)
    # Get the DataFrame from the GET parameter
    df = pd.read_html(request.GET['data'])[0]

    # Check if the DataFrame has data
    if df.empty:
        raise ValueError('DataFrame is empty')
    section_name = request.GET['section']
    file_name = f'{section_name}.xlsx'

    # Generate the Excel file and write it to a buffer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    buffer.seek(0)

    # Encode the file name using the urlencode function
    file_name = quote(file_name)

    # Create the HttpResponse object with the appropriate headers
    response = HttpResponse(buffer.read(),
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    return response


# image processing functions

# start

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# function to detect the biggest contour
def biggest_contour(contours: np.ndarray[np.int32]) -> np.ndarray[np.int32]:
    biggest = np.array([])
    max_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 1000:

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest


def find_paper(image: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    '''
        Find an answer sheet in the image and auto cropped
    '''

    # define readed answersheet image output size
    (max_width, max_height) = (827, 1669)
    # (max_width, max_height) = (830, 1800)

    img_original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)

    # binerization of the biggest contour
    cv2.drawContours(image, [biggest], -1, (0, 255, 0), 3)

    # Pixel values in the original image

    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

    return img_output


def read_answer(roi: np.ndarray[np.uint8], n_questions: int, debug: bool = True) -> list[int]:
    '''
        Read answer mark from a specific region of the answer sheet and return a result as a list.
    '''

    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize=(15, 15), sigmaX=1)

    # plt.imshow(inp, interpolation='nearest')
    # plt.show()

    (_, res) = cv2.threshold(inp, 185, 255, cv2.THRESH_BINARY)

    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=3)
    res = cv2.dilate(res, kernel=(3, 3))

    if debug:
        cv2.imshow(str(uuid4()), res)
        cv2.waitKey(0)

    (contours, _) = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    readed = []

    for cnt in contours[1:][::-1]:
        (x, y, _w, _h) = cv2.boundingRect(cnt)

        if debug:
            print(x, y)

        if x in range(0, 20):
            readed.append((int(y // 27) + 1, 1))

        elif x in range(20, 40):
            readed.append((int(y // 27) + 1, 2))

        elif x in range(40, 60):
            readed.append((int(y // 27) + 1, 3))

        elif x in range(60, 80):
            readed.append((int(y // 27) + 1, 4))

    read = [None] * n_questions

    for (n, choice) in readed:
        read[n - 1] = choice

    return read


def ans_block_read(image: np.ndarray[np.uint8], n_block: int) -> list[int]:
    '''
        Read answer from \'n\' blocks of the main answer sheet.
    '''
    # plt.imshow(image, interpolation='nearest')
    # plt.show()

    answers = []
    num_b = 0
    if n_block < 5:
        j = 0
        for i in range(0, n_block):
            j += 1
            # img = image[690 + (i * 190):845 + (i * 190), 105:190]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 95:187]

            # plt.imshow(img, interpolation='nearest')
            # plt.show()

            if j > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

    elif n_block >= 5 and n_block < 9:
        j = 0
        k = 0
        for i in range(0, n_block - 1):
            j += 1
            # img = image[690 + (i * 190):845 + (i * 190), 105:190]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 95:187]

            # plt.imshow(img, interpolation='nearest')
            # plt.show()

            # if set(read) == {None}:
            #     break
            if j > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            k += 1
            # img = image[690 + (i * 190):845 + (i * 190), 245:330]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 235:327]

            # plt.imshow(img, interpolation='nearest')
            # plt.show()

            # if set(read) == {None}:
            #     break
            if k > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

    elif n_block >= 9 and n_block < 13:
        j = 0
        k = 0
        m = 0
        for i in range(0, n_block - 1):
            j += 1
            # img = image[690 + (i * 190):845 + (i * 190), 105:190]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 95:187]

            # plt.imshow(img, interpolation='nearest')
            # plt.show()

            # if set(read) == {None}:
            #     break
            if j > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            k += 1
            # img = image[690 + (i * 190):845 + (i * 190), 242:327]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 235:327]

            #

            # if set(read) == {None}:
            #     break
            if k > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            m += 1
            # img = image[690 + (i * 190):845 + (i * 190), 382:467]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 375:467]

            #

            # if set(read) == {None}:
            #     break
            if m > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

    elif n_block >= 13 and n_block < 17:
        j = 0
        k = 0
        m = 0
        n = 0
        for i in range(0, n_block - 1):
            j += 1
            # img = image[690 + (i * 190):845 + (i * 190), 102:187]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 95:187]

            # if set(read) == {None}:
            #     break
            if j > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            k += 1
            # img = image[690 + (i * 190):845 + (i * 190), 242:327]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 235:327]

            # if set(read) == {None}:
            #     break
            if k > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            m += 1
            # img = image[690 + (i * 190):845 + (i * 190), 382:467]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 375:467]

            # if set(read) == {None}:
            #     break
            if m > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            n += 1
            # img = image[690 + (i * 190):845 + (i * 190), 382:467]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 515:607]

            # if set(read) == {None}:
            #     break
            if n > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

    elif n_block >= 17 and n_block < 21:

        j = 0
        k = 0
        l = 0
        m = 0
        n = 0
        for i in range(0, n_block - 1):
            j += 1
            # img = image[690 + (i * 190):845 + (i * 190), 105:190]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 95:187]

            # plt.imshow(img, interpolation='nearest')
            # plt.show()

            if j > 4:
                break
            # if set(read) == {None}:
            #     print('im here')
            #     answers.append(read)
            #     # break
            # else:
            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            k += 1
            # img = image[690 + (i * 190):845 + (i * 190), 245:330]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 235:327]

            if k > 4:
                break
            # if set(read) == {None}:
            #     answers.append(read)
            # break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            l += 1
            # img = image[690 + (i * 190):845 + (i * 190), 385:470]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 375:467]

            # if set(read) == {None}:
            #     break
            if l > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            m += 1
            # img = image[690 + (i * 190):845 + (i * 190), 525:610]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 515:607]

            # if set(read) == {None}:
            #     break
            if m > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

        for i in range(0, n_block - 1):
            n += 1
            # img = image[690 + (i * 190):845 + (i * 190), 665:750]
            img = image[682 + (i * 48) + (i * 150):840 + (i * 48) + (i * 150), 654:747]

            # if set(read) == {None}:
            #     break
            if n > 4:
                break

            answers.append(read_answer(img, 5, debug=False))
            num_b += 1
            if num_b == n_block:
                return [j for i in answers for j in i]

    elif n_block > 21:
        raise ValueError("n_block must be less than or equal to 20 blocks")

    return [j for i in answers for j in i]


def id_block_read(image: np.ndarray[np.uint8], debug: bool = True) -> int:
    '''
        Read the ID from the id section of the answer sheet image
    '''

    # img = image[340:625, 300:370]
    img = image[320:615, 63:280]

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = cv2.GaussianBlur(grey, ksize=(3, 3), sigmaX=1)

    (_, res) = cv2.threshold(inp, 178, 255, cv2.THRESH_BINARY)

    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=4)
    res = cv2.dilate(res, kernel=(5, 5), iterations=3)

    id_str = ''

    for i in range(1, 11):
        (contours, _) = cv2.findContours(res[:, (i - 1) * 19:i * 21 + 2], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if debug:
            cv2.imshow(str(uuid4()), res[:, (i - 1) * 21:i * 22])
            cv2.waitKey(0)

        for cnt in (contours[1:][::-1]):

            # cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
            # cv2.imshow('Contours', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if len(id_str) == 10:
                break

            (x, y, w, h) = cv2.boundingRect(cnt)
            print('this is y', y)

            if debug:
                print(y)

            if y in range(0, 20):
                id_str += '1'

            elif y in range(20, 50):
                id_str += '2'

            elif y in range(50, 80):
                id_str += '3'

            elif y in range(80, 110):
                id_str += '4'

            elif y in range(110, 130):
                id_str += '5'

            elif y in range(130, 160):
                id_str += '6'

            elif y in range(160, 190):
                id_str += '7'

            elif y in range(190, 220):
                id_str += '8'

            elif y in range(220, 250):
                id_str += '9'

            elif y in range(250, 280):
                id_str += '0'

    if id_str:
        return int(id_str)
    else:
        return 9999999999


def rotate_image(image: np.ndarray[np.uint8], angle: int) -> np.ndarray[np.uint8]:
    '''
        Rotate image for n degree.
    '''

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


# ends


def process(request, pk):
    blocks = request.GET.get('blocks')
    score = request.GET.get('score')

    section = get_object_or_404(Section, id=pk)
    # images = Image.objects.get(pk=pk)
    images = Image.objects.filter(section=section)

    for image in images:
        with open(image.image.path, 'rb') as f:
            img = PIL.Image.open(f)
            print('test', type(img))

    print(len(images))
    imlist = []
    image_list = []
    # for image in images:
    #     if image.image:
    #         imlist.append(image.image.url)

    # for path in imlist:
    #     image_list.append({'path': path})

    correct_ans = []
    datasets = []

    print('Reading answers from the sheet...')

    # for img in image_list:
    for image in images:
        with open(image.image.path, 'rb') as f:
            # image = cv2.imread(os.path.join(BASE_DIR,img['path']))
            image = cv2.imread(image.image.path)

            answer_sheet = cv2.resize(image, (827, 1669))
            student_id = id_block_read(answer_sheet, debug=False)
            answers = ans_block_read(answer_sheet, int(blocks))

            if student_id == 0:
                correct_ans = answers

            data = {'id': student_id, 'answers': answers}
            datasets.append(data.copy())

    # datasets = sorted(datasets, key=lambda data: data['id'])[1:]
    datasets = sorted(datasets, key=lambda data: data['id'])

    print(f'Correct answers(ID = 0): {correct_ans}')

    cv2.destroyAllWindows()

    for (idx, data) in enumerate(datasets):

        datasets[idx]['answers_check'] = []

        for (base, student) in zip(correct_ans, data['answers']):
            datasets[idx]['answers_check'].append(base == student)

    for data in datasets:
        print('this is data', data)
        (_, count) = np.unique(data['answers_check'], return_counts=True)
        print('this is count', (_, count))
        print('this is count 0', (_, count)[0])

        print(len(count))

        if len(count) == 2:
            # print('its count one',count[1])
            data['correct'] = count[1]
            data['incorrect'] = count[0]
        if len(count) == 1:
            if (_, count)[0][0] == False:
                data['incorrect'] = count[0]
                data['correct'] = 0
            else:
                data['correct'] = count[0]
                data['incorrect'] = 0

    for i in datasets:
        print(i['answers'])
        for j in range(len(i['answers'])):
            if i['answers'][j] == None:
                i['answers'][j] = [0, 0, 0, 0]
            if i['answers'][j] == 1:
                i['answers'][j] = [1, 0, 0, 0]
            if i['answers'][j] == 2:
                i['answers'][j] = [0, 1, 0, 0]
            if i['answers'][j] == 3:
                i['answers'][j] = [0, 0, 1, 0]
            if i['answers'][j] == 4:
                i['answers'][j] = [0, 0, 0, 1]

        i['answers'] = [i['answers'][j:j + 5] for j in range(0, len(i['answers']), 5)]

    df = pd.DataFrame(datasets)
    df['pass'] = ["Pass" if d >= int(score) else "Fail" for d in df['correct']]

    # excel = df.to_excel('out.xlsx', index=False)
    #
    # df = pd.read_excel(excel)

    print(f'Exporting data to excel spreadsheet at: {os.path.abspath("./out.xlsx")}')
    context = {'df': df[['id', 'answers']], 'section': section}
    return render(request, 'base/process.html', context)
