import cv2 
import numpy as np 
from scan.scanner import Scanner
from inference.inference import Infer


class OCR:
	def __init__(self, image: np.ndarray):
		self.image = image  

	def begin(self):
		snipped = Scanner(self.image).scan()
		Infer(snipped).start()


if __name__ == "__main__":  
	OCR(cv2.imread("C:\\Users\\moeid\\Desktop\\TXractor\\TXtractor\\OCR\\main\\scan\\images\\test_2.jpg")).begin()
