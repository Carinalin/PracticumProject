# python efficientad.py --dataset goodsad --subdataset cigarette_box --model_size medium --weights models/teacher_medium.pth > train_cb.log 2>&1
# python efficientad.py --dataset goodsad --subdataset food_box --model_size medium --weights models/teacher_medium.pth > train_fbox.log 2>&1
# python efficientad.py --dataset goodsad --subdataset drink_bottle --model_size medium --weights models/teacher_medium.pth > train_db.log 2>&1
# python efficientad.py --dataset goodsad --subdataset drink_can --model_size medium --weights models/teacher_medium.pth > train_dc.log 2>&1
# python efficientad.py --dataset goodsad --subdataset food_bottle --model_size medium --weights models/teacher_medium.pth > train_fb.log 2>&1
# python efficientad.py --dataset goodsad --subdataset food_package --model_size medium --weights models/teacher_medium.pth > train_fp.log 2>&1
# python efficientad.py --dataset goodsad --subdataset cigarette_box > train_cb.log 2>&1
# python efficientad.py --dataset goodsad --subdataset food_box > train_fbox.log 2>&1
# python efficientad.py --dataset goodsad --subdataset drink_bottle > train_db.log 2>&1
# python efficientad.py --dataset goodsad --subdataset drink_can > train_dc.log 2>&1
python efficientad.py --dataset goodsad --subdataset food_bottle > train_fb.log 2>&1
python efficientad.py --dataset goodsad --subdataset food_package > train_fp.log 2>&1