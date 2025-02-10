# Originated from https://github.com/amdegroot/ssd.pytorch
"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'OPHICHTHUSPUNCTICEPS-143150402',
    'BALISTESCAPRISCUS-189030502',
    'EPINEPHELUSMORIO-170021211',
    'PTEROIS-168011900',
    'PRIACANTHUSARENATUS-170050101',
    'PRISTIPOMOIDES-170151800',
    'SERIOLAFASCIATA-170113103',
    'RHOMBOPLITESAURORUBENS-170152001',
    'HOLACANTHUSBERMUDENSIS-170290102',
    'CHAETODONOCELLATUS-170260307',
    'MYCTEROPERCAPHENAX-170022105',
    'SYACIUM-183011000',
    'CAULOLATILUSCYANOPS-170070101',
    'LUTJANUSGRISEUS-170151109',
    'PAGRUSPAGRUS-170212302',
    'CARCHARHINUS-108020200',
    'UROPHYCISREGIA-148010105',
    'CALAMUSPRORIDENS-170210605',
    'CALAMUSNODOSUS-170210608',
    'SERIOLARIVOLIANA-170113105',
    'CALAMUS-170210600',
    'SERRANUSANNULARIS-170024201',
    'GYMNOTHORAXSAXICOLA-143060205',
    'SYNODONTIDAE-129040000',
    'IOGLOSSUS -170550800',
    'PRISTIGENYSALTA-170050401',
    'HALICHOERES-170281200',
    'CALLIONYMIDAE-170420000',
    'SERIOLADUMERILI-170113101',
    'LUTJANUSSYNAGRIS-170151113',
    'CARANXCRYSOS-170110803',
    'PSEUDUPENEUSMACULATUS-170220701',
    'GYMNOTHORAXMORINGA-143060202',
    'MALACANTHUSPLUMIERI-170070301',
    'POMACANTHUSARCUATUS-170290201',
    'LUTJANUSVIVANUS-170151114',
    'LUTJANUSCAMPECHANUS-170151107',
    'BALISTESVETULA-189030504',
    'UNKNOWNFISH',
    'MYCTEROPERCABONACI-170022101',
    'OPISTOGNATHUS-170310200',
    'MYCTEROPERCAMICROLEPIS-170022104',
    'HALICHOERESBIVITTATUS-170281202',
    'SERRANUS-170024200',
    'SERRANUSPHOEBE-170024208',
    'SERIOLAZONATA-170113106',
    'SPHYRNALEWINI-108040102',
    'MYCTEROPERCA-170022100',
    'ANISOTREMUSVIRGINICUS-170190105',
    'HYPORTHODUSNIGRITUS-170021202',
    'CARANGIDAE-170110000',
    'PRONOTOGRAMMUSMARTINICENSIS-170025101',
    'CHROMIS-170270300',
    'CHAETODONCAPISTRATUS-170260302',
    'XANTHICHTHYSRINGENS-189030101',
    'CALAMUSLEUCOSTEUS-170210604',
    'ANOMURA-999100401',
    'LACHNOLAIMUSMAXIMUS-170281801',
    'POMACENTRIDAE-170270000',
    'DECAPTERUS-170111200',
    'CARCHARHINUSPLUMBEUS-108020208',
    'GOBIIDAE-170550000',
    'MURAENARETIFERA-143060402',
    'HOLOCENTRUSADSCENSIONIS-161110201',
    'HOLOCENTRUS-161110200',
    'HALICHOERESBATHYPHILUS-170281201',
    'CALAMUSBAJONADO-170210602',
    'CARANXRUBER-170110807',
    'EPINEPHELUSADSCENSIONIS-170021203',
    'CARCHARHINUSFALCIFORMIS-108020202',
    'DIPLECTRUMFORMOSUM-170020903',
    'SERIOLA-170113100',
    'LUTJANUSBUCCANELLA-170151106',
    'HAEMULONAUROLINEATUM-170191003',
    'CENTROPRISTISOCYURUS-170024804',
    'HYPORTHODUSFLAVOLIMBATUS-170021206',
    'MYCTEROPERCAINTERSTITIALIS-170022103',
    'PARANTHIASFURCIFER-170022701',
    'LACTOPHRYSTRIGONUS-189070205',
    'HAEMULONPLUMIERI-170191008',
    'POMACENTRUSPARTITUS-170270502',
    'ACANTHURUSCOERULEUS-170160102',
    'HYPOPLECTRUSUNICOLOR-170021501',
    'SCARIDAE-170300000',
    'BODIANUSRUFUS-170280202',
    'HAEMULONMACROSTOMUM-170191017',
    'POMACENTRUS-170270500',
    'OCYURUSCHRYSURUS-170151501',
    'CEPHALOPHOLISFULVA-170020403',
    'HAEMULONFLAVOLINEATUM-170191005',
    'POMACANTHUSPARU-170290203',
    'ACANTHURUS-170160100',
    'PAREQUESUMBROSUS-170201105',
    'RYPTICUSMACULATUS-170030106',
    'LUTJANUS-170151100',
    'ARCHOSARGUSPROBATOCEPHALUS-170213601',
    'PROGNATHODESACULEATUS-170260305',
    'SCARUSVETULA-170301107',
    'POMACANTHUS-170290200',
    'HALICHOERESGARNOTI-170281205',
    'LUTJANUSAPODUS-170151102',
    'THALASSOMABIFASCIATUM-170282801',
    'SPARISOMAVIRIDE-170301206',
    'CARANXBARTHOLOMAEI-170110801',
    'HOLACANTHUS-170290100',
    'SPHYRAENABARRACUDA-165030101',
    'UPENEUSPARVUS-170220605',
    'LUTJANUSANALIS-170151101',
    'CAULOLATILUSCHRYSOPS-170070104',
    'LIOPROPOMAEUKRINES-170025602',
    'EQUETUSLANCEOLATUS-170201104',
    'HYPOPLECTRUS-170021500',
    'MULLOIDICHTHYSMARTINICUS-170220101',
    'KYPHOSUS-170240300',
    'CHAETODON-170260300',
    'SPARISOMAAUROFRENATUM-170301201',
    'STENOTOMUSCAPRINUS-170213403',
    'BODIANUSPULCHELLUS-170280201',
    'CEPHALOPHOLISCRUENTATA-170020401',
    'CHROMISINSOLATUS-170270304',
    'CHAETODONSEDENTARIUS-170260309',
    'SERRANUSATROBRANCHUS-170024202',
    'SCOMBEROMORUS-170440800',
    'DIODONTIDAE-189090000',
    'GONIOPLECTRUSHISPANUS-170021403',
    'IOGLOSSUS-170550800',
    'HYPOPLECTRUSGEMMA-170021503',
    'CANTHIGASTERROSTRATA-189080101',
    'CENTROPRISTISPHILADELPHICA-170024805',
    'RACHYCENTRONCANADUM-170100101',
    'SPARIDAE-170210000',
    'EPINEPHELUS-170021200',
    'CHROMISENCHRYSURUS-170270302',
    'CANTHIDERMISSUFFLAMEN-189030402',
    'HAEMULONMELANURUM-170191007',
    'OPISTOGNATHUSAURIFRONS-170310203',
    'DERMATOLEPISINERMIS-170020301',
    'ALECTISCILIARIS-170110101',
    'OPHICHTHUSPUNCTICEPS-143150402',
    'PROGNATHODESAYA-170260301',
    'CAULOLATILUSMICROPS-170070103',
    'HAEMULONALBUM-170191002',
    'CARCHARHINUSPEREZI-108020211',
    'DIPLECTRUM-170020900'   
)


VOC_ROOT = osp.join("/content/FATES-ATI-ActiveLearning/main_data")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            #difficult = int(obj.find('difficult').text) == 1
            #if not self.keep_difficult and difficult:
                #continue
            #name = obj.find('name').text.lower().strip()
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                cur_pt = int(cur_pt)
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]

        return res


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """


    def __init__(self, root,
                 image_sets=[('2007', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        
        #print(index)
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
