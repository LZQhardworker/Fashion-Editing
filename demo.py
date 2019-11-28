import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
import cv2
import numpy as np
from options.test_options import TestOptions
# from model import Model
import os
import time
import torch
import pix2pixHD_model as pix
from models import create_model
COUNT=0
class Ex(QWidget, Ui_Form):
    def __init__(self, model):
        super().__init__()


        self.setupUi(self)
        self.show()
        self.model = model

        self.output_img = None

        self.mat_img = None

        self.ld_mask = None
        self.ld_sk = None

        self.modes = [0,0,0]
        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            # print(fileName)
            file_label='test_label/'+fileName.split('/')[-1]
            # print(file_label)
            file_label=file_label.split('.')[0]+'.png'
            self.mat_label=cv2.imread(file_label)
            # print(self.mat_label)

            self.mat_label=cv2.cvtColor(self.mat_label,cv2.COLOR_BGR2GRAY)
            mat_img = cv2.imread(fileName)

            mat_img=cv2.cvtColor(mat_img,cv2.COLOR_BGR2RGB)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return

            # redbrush = QBrush(Qt.red)
            # blackpen = QPen(Qt.black)
            # blackpen.setWidth(5)
            self.image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            mat_img = mat_img/127.5 - 1
            self.mat_img = np.expand_dims(mat_img,axis=0)
            # cv2.imshow('a',self.mat_img.squeeze())
            # cv2.waitKey(0)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(self.image)

    def mask_mode(self):
        self.mode_select(0)

    def sketch_mode(self):
        self.mode_select(1)

    def stroke_mode(self):
        if not self.color:
            self.color_change_mode()
        self.scene.get_stk_color(self.color)
        self.mode_select(2)

    def color_change_mode(self):
        self.dlg.exec_()
        self.color = self.dlg.currentColor().name()
        self.pushButton_4.setStyleSheet("background-color: %s;" % self.color)
        self.scene.get_stk_color(self.color)

    def complete(self):
        global COUNT
        sketch = self.make_sketch(self.scene.sketch_points)
        sketch=torch.FloatTensor(sketch)
        stroke = self.make_stroke(self.scene.stroke_points)
        stroke=torch.FloatTensor(stroke)
        mask = self.make_mask(self.scene.mask_points)
        mask=torch.FloatTensor(mask)
        self.mat_img=torch.FloatTensor(self.mat_img)
        if not type(self.ld_mask)==type(None):
            ld_mask = np.expand_dims(self.ld_mask[:,:,0:1],axis=0)
            ld_mask[ld_mask>0] = 1
            ld_mask[ld_mask<1] = 0
            mask = mask+ld_mask
            mask[mask>0] = 1
            mask[mask<1] = 0
            mask = np.asarray(mask,dtype=np.uint8)

        if not type(self.ld_sk)==type(None):
            sketch = sketch+self.ld_sk
            sketch[sketch>0]=1 

        noise = self.make_noise()
        noise=torch.FloatTensor(noise)
        self.mat_label=torch.FloatTensor(self.mat_label)
        self.mat_label=self.mat_label.reshape(1,512,320,1)

        start_t = time.time()
        result = self.model.inference(self.mat_label.permute(0,3,1,2).cuda()
                                      ,self.mat_img.permute(0,3,1,2).cuda()
                                      ,sketch.permute(0,3,1,2).cuda()
                                      ,stroke.permute(0,3,1,2).cuda()
                                      ,mask.permute(0,3,1,2).cuda()
                                      ,noise.permute(0,3,1,2).cuda())
        result=result*mask.permute(0,3,1,2).cuda()+self.mat_img.permute(0,3,1,2).cuda()*(1-mask.permute(0,3,1,2).cuda())
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        result = (result+1)*127.5
        result=result.permute(0,2,3,1)
        result=result.cpu().numpy()
        result = np.asarray(result[0,:,:,:],dtype=np.uint8)

        self.output_img = result
        qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
        self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))
        COUNT+=1

    def make_noise(self):
        noise = np.zeros([512, 320, 1],dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise/255,dtype=np.uint8)
        noise = np.expand_dims(noise,axis=0)
        return noise

    def make_mask(self, pts):
        if len(pts)>0:
            mask = np.zeros((512,320,3))
            for pt in pts:
                cv2.line(mask,pt['prev'],pt['curr'],(255,255,255),12)
            mask = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=0)
        else:
            mask = np.zeros((512,320,3))
            mask = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=0)
        return mask

    def make_sketch(self, pts):
        if len(pts)>0:
            sketch = np.zeros((512,320,3))
            # sketch = 255*sketch
            for pt in pts:
                cv2.line(sketch,pt['prev'],pt['curr'],(255,255,255),1)
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        else:
            sketch = np.zeros((512,320,3))
            # sketch = 255*sketch
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        return sketch

    def make_stroke(self, pts):
        if len(pts)>0:
            stroke = np.zeros((512,320,3))
            for pt in pts:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2 ,4))
                cv2.line(stroke,pt['prev'],pt['curr'],color,4)
            stroke = stroke/127.5 - 1
            stroke = np.expand_dims(stroke,axis=0)
        else:
            stroke = np.zeros((512,320,3))
            stroke = stroke/127.5 - 1
            stroke = np.expand_dims(stroke,axis=0)
        return stroke

    def arrange(self):
        image = np.asarray((self.mat_img[0]+1)*127.5,dtype=np.uint8)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if len(self.scene.mask_points)>0:
            for pt in self.scene.mask_points:
                cv2.line(image,pt['prev'],pt['curr'],(255,255,255),12)
        if len(self.scene.stroke_points)>0:
            for pt in self.scene.stroke_points:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2 ,4))
                color = (color[2],color[1],color[0])
                cv2.line(image,pt['prev'],pt['curr'],color,4)
        if len(self.scene.sketch_points)>0:
            for pt in self.scene.sketch_points:
                cv2.line(image,pt['prev'],pt['curr'],(0,0,0),1)        
        cv2.imwrite('tmp.jpg',image)
        image = QPixmap('tmp.jpg')
        self.scene.history.append(3)
        self.scene.addPixmap(image)

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                    QDir.currentPath())
            cv2.imwrite(fileName+'.jpg',self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)

if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    model = create_model(opt)
    app = QApplication(sys.argv)
    ex = Ex(model)
    sys.exit(app.exec_())
