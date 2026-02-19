# **IKEA-Gen: Specialized Interior Design via ControlNet & LoRA**

### <img width="1259" height="605" alt="image" src="https://github.com/user-attachments/assets/096844cd-8145-4189-9ca5-770fb3119724" />



---

## **Project Overview**

This research project focuses on the domain-specific adaptation of **Stable Diffusion v1.5** for high-fidelity IKEA interior generation. By combining **ControlNet (Canny)** for structural guidance with **LoRA (Low-Rank Adaptation)** for aesthetic specialization, we developed a pipeline capable of transforming simple architectural sketches into professional-grade minimalist designs.

## **Technical Collaboration & Roles**

To simulate a professional ML research environment, the project was divided into two core phases:

### **Phase 1: Dataset Engineering (Lead: Gourab Bhadra)**

* **Notebook:** `IKEA_Dataset_Engineering_Pipeline.ipynb`
* **Scope:** Architected a robust data pipeline to curate and preprocess **2,530 high-resolution image pairs**.
* **Key Contributions:**
* Implemented automated **Canny Edge Detection** to generate structural conditioning maps.
* Developed a metadata management system using **JSONL** for seamless integration with the Hugging Face `datasets` library.
* Managed persistent **Google Drive-to-Colab** cloud storage to ensure data integrity during large-scale processing.



### **Phase 2: Model Fine-Tuning & Optimization (Lead: Anubhab Dutta)**

* **Notebook:** `ControlNet_LoRA_FineTuning_Inference.ipynb`
* **Scope:** Specialized the foundation model's textures and lighting using **Parameter-Efficient Fine-Tuning (PEFT)**.
* **Key Contributions:**
* Injected **LoRA adapters** into the U-Net cross-attention layers, reducing trainable parameters by over **90%** while maintaining model stability.
* Optimized the training loop on a **Tesla T4 GPU** using **Mixed-Precision (FP16)** and a **Cosine Learning Rate Scheduler**.
* Developed an inference workflow utilizing the **UniPCMultistepScheduler** to achieve high-quality results in only 30-50 steps.



## **Technical Challenges Overcome**

* **Meta-Tensor Mismatches:** Resolved `NotImplementedError` issues by implementing a modular loading strategy, ensuring weights were correctly mapped to GPU memory before inference.
* **Resource Constraints:** Managed the **16GB VRAM limit** of the Tesla T4 through efficient gradient management and model offloading.
* **Structural Fidelity:** Tuned the `controlnet_conditioning_scale` to **0.8** to balance architectural accuracy with realistic lighting and texture generation.

---

## **How to Run**

1. **Clear All Outputs:** Ensure you clear notebook outputs before running to avoid metadata widget errors.
2. **Environment:** Recommended for **Google Colab** with a T4 GPU or higher.
3. **Requirements:**
```bash
pip install -r requirements.txt

```
