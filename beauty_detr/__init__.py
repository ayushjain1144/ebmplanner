# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2021 Ayush Jain & Nikolaos Gkanatsios
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Modified from MDETR 
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------


from .bdetr import build_bdetr

def build_bdetr_model(args):
    return build_bdetr(args)
