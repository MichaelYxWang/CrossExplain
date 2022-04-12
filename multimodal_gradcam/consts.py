class global_consts():
    data_path = "../Dataset/"
    dataset = "hateful_memes"
    model_dir = "./model/"
    load_model = False
    save_model = True

    if dataset == "hateful_memes":
        text_pretrained_model = 'bert-base-cased'
        dropout = 0.3
        batch_size = 1
        class_num = 1
        lr = 0.001
        epoch_num = 10
        filter_image_list = ['img/80567.png', 'img/49270.png', 'img/58306.png', 'img/17832.png', 'img/17205.png', 'img/75428.png', 'img/42015.png', 'img/38624.png', 'img/30847.png', 'img/07312.png', 'img/74589.png', 'img/86159.png', 'img/51304.png', 'img/18296.png', 'img/69807.png', 'img/65908.png', 'img/65940.png', 'img/79854.png', 'img/93124.png', 'img/36725.png', 'img/94802.png', 'img/21043.png', 'img/09478.png', 'img/73984.png', 'img/59248.png', 'img/98412.png', 'img/26543.png', 'img/96701.png', 'img/86104.png', 'img/48579.png', 'img/84076.png', 'img/42073.png', 'img/98367.png', 'img/56193.png', 'img/41823.png', 'img/85269.png', 'img/16702.png', 'img/82156.png', 'img/90483.png', 'img/40158.png', 'img/17928.png', 'img/32781.png', 'img/58194.png', 'img/82540.png', 'img/36178.png', 'img/94150.png', 'img/16084.png', 'img/95817.png', 'img/16438.png', 'img/85147.png', 'img/65904.png', 'img/85392.png', 'img/43791.png', 'img/69025.png', 'img/79423.png', 'img/49067.png', 'img/84951.png', 'img/25310.png', 'img/23174.png', 'img/96340.png', 'img/98736.png', 'img/08741.png', 'img/10835.png', 'img/48192.png', 'img/40917.png']
