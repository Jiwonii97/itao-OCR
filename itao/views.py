from rest_framework.response import Response
from rest_framework import status

from rest_framework.decorators import api_view
from .color import image_color_cluster

import numpy as np
from PIL import Image
import io
import base64
import boto3


from dotenv import load_dotenv
import os

load_dotenv()   # load .env
ACCESS_ID = os.environ.get("ACCESS_ID")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
REGION = os.environ.get("REGION")

client = boto3.client('textract',
                      region_name=REGION,
                      aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=ACCESS_KEY)


@api_view(["POST"])
def myTranslate(request):
    try:
        image = request.data['image']
        data = base64.b64decode(image)

        img = Image.open(io.BytesIO(data))
        img = np.asarray(img)

        textBox = client.detect_document_text(Document={'Bytes': data})
        colorBox = image_color_cluster(img)

        return Response({
            "color": colorBox,
            "text": textBox
        }, status=status.HTTP_200_OK)

    except ValueError as e:
        print(e)
        return Response(status.HTTP_400_BAD_REQUEST, status=status.HTTP_400_BAD_REQUEST)
