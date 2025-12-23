# 🧠 PLS4MIS: Partially Labeled Supervision for Medical Image Segmentation

**PLS4MIS** is an open-source toolbox for **partially labeled medical image segmentation**.

* This project aims to facilitate research in scenarios where full pixel-wise annotations are expensive or infeasible by providing literature reviews, benchmark implementations, and practical PyTorch code.

* This project was originally developed for our previous works. We are continuing to extend it to be more user-friendly and to support additional approaches that further facilitate research in this area. **If you use this codebase in your research, please cite the following works**:

        @article{li2025pl,
        title={PL-Seg: Partially labeled abdominal organ segmentation via classwise orthogonal contrastive learning and progressive self-distillation},
        author={Li, He and Luo, Xiangde and Fu, Jia and Gu, Ran and Liao, Wenjun and Zhang, Shichuan and Li, Kang and Wang, Guotai and Zhang, Shaoting},
        journal={Medical Image Analysis},
        pages={103885},
        year={2025},
        publisher={Elsevier}}

---

## 📌 Highlights

- 📁 Focused on partially labeled supervision for **3D medical image segmentation**
- 📚 Includes **daily-updated literature reviews**
- 🛠️ Implements **six representative algorithms**
- 🧪 Ready-to-run examples and scripts

---

## 📊 Datasets for partially labeled medical image segmentation.
Some information and download links of the partially labeled learning datasets can be found in this [Link](https://github.com/HiLab-git/PLS4MIS/tree/main/datasets).

---

## 🔬 Code for partially labeled medical image segmentation.
Some implementations of partially labeled learning methods can be found in this [Link](https://github.com/HiLab-git/PLS4MIS/tree/main/code).

---

## 📖 Literature reviews of partially labeled learning approach for medical image segmentation (**PLS4MIS**)
|Date|The First and Last Authors|Title|Code|Reference|
|---|---|---|---|---|
|2025-10|Z. Zhang and X. Duan|AMOTS: Partially supervised framework for abdominal multi-organ and tumor segmentation via aspect-aware complementary|[Code](https://github.com/zzm3zz/AMOTS)|[AIMed2025](https://www.sciencedirect.com/science/article/pii/S0933365725001599?ref=pdf_download&fr=RR-2&rr=966ba42cac9fcbae)|
|2025-09|X. Liu and Z. Song|Deep Mutual Learning among Partially Labeled Datasets for Multi-Organ Segmentation|None|[TMI2025](https://ieeexplore.ieee.org/abstract/document/11181137)|
|2025-09|S. Zhu and J. Hu|Visual prompt-driven universal model for medical image segmentation in radiotherapy|None|[KBS2025](https://www.sciencedirect.com/science/article/pii/S0950705125010512)|
|2025-07|H. Gong and H. Li|Boundary as the Bridge: Toward Heterogeneous Partially-Labeled Medical Image Segmentation and Landmark Detection|[Code](https://github.com/lhaof/HPL)|[TMI2025](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10915612)|
|2025-01|X. Jiang and X. Yang|Labeled-to-unlabeled distribution alignment for partially-supervised multi-organ medical image segmentation|[Code](https://github.com/xjiangmed/LTUDA)|[MedIA2025](https://www.sciencedirect.com/science/article/pii/S1361841524002585)|
|2024-11|Q. Liu and Y. Liang|Many birds, one stone: Medical image segmentation with multiple partially labeled datasets|[Code](https://github.com/CVIU-CSU/PSSNet)|[PR2024](https://www.sciencedirect.com/science/article/pii/S003132032400387X)|
|2024-10|J. Liu and Z. Zhou|Universal and extensible language-vision models for organ segmentation and tumor detection from abdominal computed tomography|[Code](https://github.com/ljwztc/CLIP-Driven-Universal-Model)|[MedIA2024](https://www.sciencedirect.com/science/article/pii/S1361841524001518)|
|2024-06|B. Billot and P. Golland|Network conditioning for synergistic learning on partial annotations|[Code](https://github.com/BBillot/CoNeMOS)|[MIDL2024](https://openreview.net/forum?id=sfjgmuvLS7)|
|2024-05|H. Liu and S. Grbic|COSST: Multi-Organ Segmentation With Partially Labeled Datasets Using Comprehensive Supervisions and Self-Training|None|[TMI2024](https://ieeexplore.ieee.org/abstract/document/10400525)|
|2024-03|Y. Gao and DN. Metaxas|Training like a medical resident: Context-prior learning toward universal medical image segmentation|[Code](https://github.com/yhygao/universal-medical-image-segmentation)|[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Training_Like_a_Medical_Resident_Context-Prior_Learning_Toward_Universal_Medical_CVPR_2024_paper.html)|
|2024-03|X. Chen and Y. Fan|Versatile medical image segmentation learned from multi-source datasets via model self-disambiguation|None|[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Versatile_Medical_Image_Segmentation_Learned_from_Multi-Source_Datasets_via_Model_CVPR_2024_paper.pdf)|
|2024-02|H. Wang and S. Wan|A multi-objective segmentation method for chest X-rays based on collaborative learning from multiple partially annotated datasets|None|[InfFusion2024](https://www.sciencedirect.com/science/article/pii/S1566253523003329)|
|2023-10|Y. Ye and Y. Xia|Uniseg: A prompt-driven universal segmentation model as well as a strong representation learner|[Code](https://github.com/yeerwen/UniSeg)|[MICCAI2023](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_49)|
|2023-10|C. Ulrich and KH. Maier-Hein|MultiTalent: A Multi-dataset Approach to Medical Image Segmentation|[Code](https://github.com/MIC-DKFZ/MultiTalent)|[MICCAI2023](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_62)|
|2023-09|Y. Xie and C. Shen|Learning From Partially Labeled Data for Multi-Organ and Tumor Segmentation|[Code](https://github.com/jianpengz/DoDNet/tree/main/TransDoD)|[TPAMI2023](https://ieeexplore.ieee.org/abstract/document/10242007)|
|2023-09|R. Deng and Y. Huo|Omni-seg: A scale-aware dynamic network for renal pathological image segmentation|[Code](https://github.com/ddrrnn123/Omni-Seg)|[TBME2023](https://ieeexplore.ieee.org/abstract/document/10079171)|
|2023-06|X. Liu and S. Yang|CCQ: Cross-Class Query Network for Partially Labeled Organ Segmentation|[Code](https://github.com/Yang-007/CCQ)|[AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25264)|
|2022-08|R. Deng and Y. Huo|Omni-Seg: A Single Dynamic Network for Multi-label Renal Pathology Image Segmentation using Partially Labeled Data|[Code](https://github.com/ddrrnn123/Omni-Seg)|[MIDL2022](https://proceedings.mlr.press/v172/deng22a/deng22a.pdf)|
|2022-04|H. Wu and A. Sowmya|Tgnet: A Task-Guided Network Architecture for Multi-Organ and Tumour Segmentation from Partially Labelled Datasets|None|[ISBI2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9761582)|
|2021-09|L. Fidon and T. Vercauteren|Label-Set Loss Functions for Partial Supervision: Application to Fetal Brain 3D MRI Parcellation|[Code](https://github.com/LucasFidon/label-set-loss-functions)|[MICCAI2021](https://link.springer.com/content/pdf/10.1007/978-3-030-87196-3_60.pdf?pdf=inline%20link)|
|2021-05|G. Shi and SK. Zhou|Marginal loss and exclusion loss for partially supervised multi-organ segmentation|[Code](https://github.com/MIRACLE-Center/Partially-supervised-multi-organ-segmentation)|[MedIA2021](https://www.sciencedirect.com/science/article/pii/S1361841521000256)|
|2021-03|J. Zhang and C. Shen|DoDNet: Learning To Segment Multi-Organ and Tumors From Multiple Partially Labeled Datasets|[Code](https://github.com/jianpengz/DoDNet)|[CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_DoDNet_Learning_To_Segment_Multi-Organ_and_Tumors_From_Multiple_Partially_CVPR_2021_paper.html)|
|2020-11|X. Fang and P. Yan|Multi-Organ Segmentation Over Partially Labeled Datasets With Multi-Scale Feature Abstraction|[Code](https://github.com/DIAL-RPI/PIPO-FAN)|[TMI2020](https://ieeexplore.ieee.org/abstract/document/9112221)|
|2020-09|R. Huang and H. Li|Multi-organ segmentation via co-training weight-averaged models from few-organ datasets|None|[MICCAI2020](https://link.springer.com/chapter/10.1007/978-3-030-59719-1_15)|
|2019-11|Y. Zhou and AL. Yuille|Prior-Aware Neural Network for Partially-Supervised Multi-Organ Segmentation|None|[ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Prior-Aware_Neural_Network_for_Partially-Supervised_Multi-Organ_Segmentation_ICCV_2019_paper.html)|
|2019-06|K. Dmitriev and AE. Kaufman|Learning multi-class segmentations from single-class datasets|None|[CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Dmitriev_Learning_Multi-Class_Segmentations_From_Single-Class_Datasets_CVPR_2019_paper.html)|

---

## ❓ Questions and Suggestions
We welcome contributions, suggestions, and collaborations!
- 📧 Email: lihe200203@gmail.com
- 💬 QQ Group (Chinese): 906808850
