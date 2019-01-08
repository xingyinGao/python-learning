SLIC调用方式：
	第一步：
		sl.SLIC(image,step=20)
		sl.main()
	或：
		sl.rgb2Lab():RGB转Lab空间
    		sl.seeds_init()：种子点初始计算；
    		sl.seeds_adjust()：种子点梯度调整；
		sl.Seeds_grow():超像素分割
	第二步：
		rg=Region_grow(sl.dec_mat,count=100):sl.dec_mat为超像素图像,count为最							小超像素像元集合;
		rg.main()
	或：
		rg.grow()：区域增长法确定满足要求的超像素个体（如最小超像素像元集合）
        	rg.enhance():对于为满足要求像元个体重新分配像元标签（就近原则）
	第三步：（可选）
		rg.makeboundary(template_img):绘制超像素边界函数，template_img为模板						图像;