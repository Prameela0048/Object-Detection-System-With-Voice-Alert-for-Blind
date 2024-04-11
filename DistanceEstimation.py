def SWAB():
    import cv2 as cv 
    import numpy as np

    # Distance constants 
    KNOWN_DISTANCE = 45 #INCHES
    PERSON_WIDTH = 16 #INCHES
    MOBILE_WIDTH = 2.5 #INCHES
    BOTTLE_WIDTH=3.0
    BOOK_WIDTH=6.5
    CHAIR_WIDTH=17
    LAPTOP_WIDTH=14
    APPLE_WIDTH=2
    TEDDY_WIDTH=12
    MOTOR_BIKE_WIDTH=60
    CUP_WIDTH=4
    VASE_WIDTH=6
    POTTEDPLANT_WIDTH=12
    SPORTS_BALL_WIDTH=9
    CAR_WIDTH=196
    DOG_WIDTH=13
    STOPSIGN_WIDTH=30
    FIREHYDRANT_WIDTH=4.5
    TRAFFICLIGHT_WIDTH=9.5
    KEYBOARD_WIDTH=66
    REMOTE_WIDTH=2
    MOUSE_WIDTH=5
    BED_WIDTH=59
    BICYCLE_WIDTH=43
    BUS_WIDTH=90
    TRAIN_WIDTH=60
    TRUCK_WIDTH=80.4
    CAT_WIDTH=5.9
    CAR_WIDTH=70
    BUS_WIDTH=96
    BENCH_WIDTH=45
    BOAT_WIDTH=216
    PARKING_METER_WIDTH=20
    HORSE_WIDTH=56
    SHEEP_WIDTH=17
    COW_WIDTH=23
    ELEPHANT_WIDTH=82
    BEAR_WIDTH=52
    ZEBRA_WIDTH=64
    GIRAFFE_WIDTH=36
    UMBRELLA_WIDTH=43
    HANDBAG_WIDTH=4.7
    SUITCASE_WIDTH=14
    FRISBEE_WIDTH=10.5
    SNOWBOARD_WIDTH=57
    BASEBALL_BAT_WIDTH=34
    BASEBALL_GLOVE_WIDTH=9
    SKATEBOARD_WIDTH=8
    SURFBOARD_WIDTH=20
    TENNIS_RACKET_WIDTH=10.5
    WINEGLASS_WIDTH=3.5
    KNIFE_WIDTH=8



    # Object detector constant 
    CONFIDENCE_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.3


    # colors for object detected
    COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    GREEN =(0,255,0)
    BLACK =(0,0,0)
    # defining fonts 
    FONTS = cv.FONT_HERSHEY_COMPLEX

    # getting class names from classes.txt file 
    class_names = []
    with open("classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
        #  setttng up opencv net
    yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

    yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(yoloNet)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    # object detector funciton /method
    def object_detector(image):
        classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        # creating empty list to add objects data
        data_list =[]
        for (classid, score, box) in zip(classes, scores, boxes):
            # define color of each, object based on its class id 
            color= COLORS[int(classid) % len(COLORS)]

            label = "%s : %f" % (class_names[classid], score)

            # draw rectangle on and label on object
            cv.rectangle(image, box, color, 2)
            cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)

            # getting the data 
            # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
            if classid ==0: # person class id 
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==67:#mobile
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==39:#bottle
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==73:#book
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==56:#chair
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==63:#laptop\
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==47:#apple
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==77:#teddy
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==3:#BIKE
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==41:#cup
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==75:#vase
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==58:#pottedplant
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==32:#sports ball
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==16:#DOG
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==11:#stopsign
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==10:#firehydrant
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==9:#trafficlight
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==66:#keyboard
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==65:#remote
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==64:#mouse
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==59:#bed
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==1:#bicycle
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==5:#bus
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==6:#train
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==7:#truck
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==15:#cat
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==2:#car
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==5:#bus
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==13:#bench
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==8:#boat
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==12:#parking meter
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==17:#horse
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==18:#sheep
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==19:#caw
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==20:#elephant
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==21:#bear
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==22:#zebra
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==23:#giraffe
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==25:#umbrella
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==26:#handbag
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==28:#suitcase
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==29:#frisbee
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==31:#snowboard
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==34:#baseball bat
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==35:#baseball glove
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==36:#skateboard
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==37:#surfboard
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==38:#tennies racket
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==40:#wine glass
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            elif classid ==43:#knife
                data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
            
            
            
            
            #if you want inclulde more classes then you have to simply add more [elif] statements here
            # returning list containing the object data. 
        return data_list

    def focal_length_finder (measured_distance, real_width, width_in_rf):
        focal_length = (width_in_rf * measured_distance) / real_width

        return focal_length

    # distance finder function 
    def distance_finder(focal_length, real_object_width, width_in_frmae):
        distance = (real_object_width * focal_length) / width_in_frmae
        return distance

    # reading the reference image from dir 
    ref_person = cv.imread('ReferenceImages/persons.png')
    ref_mobile = cv.imread('ReferenceImages/image4.png')
    ref_bottle = cv.imread('ReferenceImages/bottle2.png')
    ref_book = cv.imread('ReferenceImages/books.png')
    ref_chair= cv.imread('ReferenceImages/chairs.png')
    ref_laptop= cv.imread('ReferenceImages/laptops.png')
    ref_apple= cv.imread('ReferenceImages/apples.png')
    ref_teddy= cv.imread('ReferenceImages/teddy.png')
    ref_motorbike= cv.imread('ReferenceImages/bike.png')
    ref_cup= cv.imread('ReferenceImages/cups.png')
    ref_vase= cv.imread('ReferenceImages/vase.png')
    ref_pottedplant= cv.imread('ReferenceImages/pottedplant.png')
    ref_sports_ball= cv.imread('ReferenceImages/sports ball.png')
    ref_car= cv.imread('ReferenceImages/car.png')
    ref_dog= cv.imread('ReferenceImages/dogs.png')
    ref_stopsign= cv.imread('ReferenceImages/stopsign.png')
    ref_firehydrant= cv.imread('ReferenceImages/fire-hydrant.png')
    ref_trafficlight= cv.imread('ReferenceImages/trafficlight.png')
    ref_keyboard= cv.imread('ReferenceImages/keyboard.png')
    ref_remote= cv.imread('ReferenceImages/remote.png')
    ref_mouse= cv.imread('ReferenceImages/mouse.png')
    ref_bed= cv.imread('ReferenceImages/bed.png')
    ref_bicycle= cv.imread('ReferenceImages/bicycle.png')
    ref_bus= cv.imread('ReferenceImages/bus.png')
    ref_train= cv.imread('ReferenceImages/train.png')
    ref_truck= cv.imread('ReferenceImages/truck.png')
    ref_cat= cv.imread('ReferenceImages/cat.png')
    ref_bus= cv.imread('ReferenceImages/buses.png')
    ref_bench= cv.imread('ReferenceImages/bench.png')
    ref_boat= cv.imread('ReferenceImages/boat.png')
    ref_parking_meter= cv.imread('ReferenceImages/parking_meter.png')
    ref_horse= cv.imread('ReferenceImages/horse.png')
    ref_sheep= cv.imread('ReferenceImages/sheep.png')
    ref_cow= cv.imread('ReferenceImages/cow.png')
    ref_elephant= cv.imread('ReferenceImages/elephant.png')
    ref_bear= cv.imread('ReferenceImages/bear.png')
    ref_zebra= cv.imread('ReferenceImages/zebra.png')
    ref_giraffe= cv.imread('ReferenceImages/giraffe.png')
    ref_umbrella= cv.imread('ReferenceImages/umbrella.png')
    ref_handbag= cv.imread('ReferenceImages/handbag.png')
    ref_suitcase= cv.imread('ReferenceImages/suitcase.png')
    ref_frisbee= cv.imread('ReferenceImages/frisbee.png')
    ref_snowboard= cv.imread('ReferenceImages/snowboard.png')
    ref_baseball_bat= cv.imread('ReferenceImages/bassball_bat.png')
    ref_baseball_glove= cv.imread('ReferenceImages/baseball_glove.png')
    ref_skateboard= cv.imread('ReferenceImages/skateboard.png')
    ref_surfboard= cv.imread('ReferenceImages/surfboard.png')
    ref_tennis_racket= cv.imread('ReferenceImages/tennis.png')
    ref_wineglass= cv.imread('ReferenceImages/wineglass.png')
    ref_knife= cv.imread('ReferenceImages/knife.png')


    person_data = object_detector(ref_person)
    person_width_in_rf = person_data[0][1] 

    mobile_data = object_detector(ref_mobile)
    mobile_width_in_rf = mobile_data[0][1]



    bottle_data = object_detector(ref_bottle)
    bottle_width_in_rf = bottle_data[0][1]

    book_data = object_detector(ref_book)
    book_width_in_rf = book_data[0][1]

    chair_data = object_detector(ref_chair)
    chair_width_in_rf = chair_data[0][1]

    laptop_data = object_detector(ref_laptop)
    laptop_width_in_rf = laptop_data[0][1]

    apple_data = object_detector(ref_apple)
    apple_width_in_rf = apple_data[0][1]

    teddy_data = object_detector(ref_teddy)
    teddy_width_in_rf = teddy_data[0][1]

    motorbike_data = object_detector(ref_motorbike)
    motorbike_width_in_rf = motorbike_data[0][1]

    cup_data = object_detector(ref_cup)
    cup_width_in_rf = cup_data[0][1]

    vase_data = object_detector(ref_vase)
    vase_width_in_rf = vase_data[0][1]

    pottedplant_data = object_detector(ref_pottedplant)
    pottedplant_width_in_rf = pottedplant_data[0][1]

    sports_ball_data = object_detector(ref_sports_ball)
    sports_ball_width_in_rf = sports_ball_data[0][1]

    car_data = object_detector(ref_car)
    car_width_in_rf = car_data[0][1]

    dog_data = object_detector(ref_dog)
    dog_width_in_rf = dog_data[0][1]

    stopsign_data = object_detector(ref_stopsign)
    stopsign_width_in_rf = stopsign_data[0][1]

    firehydrant_data = object_detector(ref_firehydrant)
    firehydrant_width_in_rf = firehydrant_data[0][1]

    trafficlight_data = object_detector(ref_trafficlight)
    trafficlight_width_in_rf = trafficlight_data[0][1]

    keyboard_data = object_detector(ref_keyboard)
    keyboard_width_in_rf = keyboard_data[0][1]

    remote_data = object_detector(ref_remote)
    remote_width_in_rf = remote_data[0][1]

    #mouse_data = object_detector(ref_mouse)
    #mouse_width_in_rf = mouse_data[0][1]

    bed_data = object_detector(ref_bed)
    bed_width_in_rf = bed_data[0][1]

    bicycle_data = object_detector(ref_bicycle)
    bicycle_width_in_rf = bicycle_data[0][1]

    bus_data = object_detector(ref_bus)
    bus_width_in_rf = bus_data[0][1]

    #train_data = object_detector(ref_train)
    #train_width_in_rf = train_data[0][1]

    truck_data = object_detector(ref_truck)
    truck_width_in_rf = truck_data[0][1]

    cat_data = object_detector(ref_cat)
    cat_width_in_rf = cat_data[0][1]

    car_data = object_detector(ref_car)
    car_width_in_rf = car_data[0][1]

    #bus_data = object_detector(ref_bus)
    #bus_width_in_rf = bus_data[0][1]

    #bench_data = object_detector(ref_bench)
    #bench_width_in_rf = bench_data[0][1]

    #boat_data = object_detector(ref_boat)
    #boat_width_in_rf = boat_data[0][1]

    parking_meter_data = object_detector(ref_parking_meter)
    parking_meter_width_in_rf = parking_meter_data[0][1]

    #horse_data = object_detector(ref_horse)
    #horse_width_in_rf = horse_data[0][1]

    sheep_data = object_detector(ref_sheep)
    sheep_width_in_rf = sheep_data[0][1]

    cow_data = object_detector(ref_cow)
    cow_width_in_rf = cow_data[0][1]

    elephant_data = object_detector(ref_elephant)
    elephant_width_in_rf = elephant_data[0][1]

    bear_data = object_detector(ref_bear)
    bear_width_in_rf = bear_data[0][1]

    zebra_data = object_detector(ref_zebra)
    zebra_width_in_rf = zebra_data[0][1]

    giraffe_data = object_detector(ref_giraffe)
    giraffe_width_in_rf = giraffe_data[0][1]

    #umbrella_data = object_detector(ref_umbrella)
    #umbrella_width_in_rf = umbrella_data[0][1]

    #handbag_data = object_detector(ref_handbag)
    #handbag_width_in_rf = handbag_data[0][1]

    #suitcase_data = object_detector(ref_suitcase)
    #suitcase_width_in_rf = suitcase_data[0][1]

    #frisbee_data = object_detector(ref_frisbee)
    #frisbee_width_in_rf = frisbee_data[0][1]

    #snowboard_data = object_detector(ref_snowboard)
    #snowboard_width_in_rf = snowboard_data[0][1]

    #baseball_bat_data = object_detector(ref_baseball_bat)
    #baseball_bat_width_in_rf = baseball_bat_data[0][1]

    #baseball_glove_data = object_detector(ref_baseball_glove)
    #baseball_glove_width_in_rf = baseball_glove_data[0][1]

    #skateboard_data = object_detector(ref_skateboard)
    #skateboard_width_in_rf = skateboard_data[0][1]

    surfboard_data = object_detector(ref_surfboard)
    surfboard_width_in_rf = surfboard_data[0][1]

    #tennis_racket_data = object_detector(ref_tennis_racket)
    #tennis_racket_width_in_rf = tennis_racket_data[0][1]

    wineglass_data = object_detector(ref_wineglass)
    wineglass_width_in_rf = wineglass_data[0][1]

    #knife_data = object_detector(ref_knife)
    #knife_width_in_rf = knife_data[0][1]

    print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

    # finding focal length 
    focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
    focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_width_in_rf)
    focal_book = focal_length_finder(KNOWN_DISTANCE, BOOK_WIDTH, book_width_in_rf)
    focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
    focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)
    focal_apple = focal_length_finder(KNOWN_DISTANCE, APPLE_WIDTH, apple_width_in_rf)
    focal_teddy = focal_length_finder(KNOWN_DISTANCE, TEDDY_WIDTH, teddy_width_in_rf)
    focal_bike = focal_length_finder(KNOWN_DISTANCE, MOTOR_BIKE_WIDTH, motorbike_width_in_rf)
    focal_cup = focal_length_finder(KNOWN_DISTANCE, CUP_WIDTH, cup_width_in_rf)
    focal_vase = focal_length_finder(KNOWN_DISTANCE, VASE_WIDTH, vase_width_in_rf)
    focal_pottedplant= focal_length_finder(KNOWN_DISTANCE, POTTEDPLANT_WIDTH, pottedplant_width_in_rf)
    focal_sports_ball= focal_length_finder(KNOWN_DISTANCE, SPORTS_BALL_WIDTH, sports_ball_width_in_rf)
    focal_car= focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)
    focal_dog= focal_length_finder(KNOWN_DISTANCE, DOG_WIDTH, dog_width_in_rf)
    focal_stopsign= focal_length_finder(KNOWN_DISTANCE, STOPSIGN_WIDTH, stopsign_width_in_rf)
    focal_firehydrant= focal_length_finder(KNOWN_DISTANCE, FIREHYDRANT_WIDTH, firehydrant_width_in_rf)
    focal_trafficlight= focal_length_finder(KNOWN_DISTANCE, TRAFFICLIGHT_WIDTH, trafficlight_width_in_rf)
    focal_keyboard= focal_length_finder(KNOWN_DISTANCE, KEYBOARD_WIDTH, keyboard_width_in_rf)
    focal_remote= focal_length_finder(KNOWN_DISTANCE, REMOTE_WIDTH, remote_width_in_rf)
    #focal_mouse= focal_length_finder(KNOWN_DISTANCE, MOUSE_WIDTH, mouse_width_in_rf)
    focal_bed= focal_length_finder(KNOWN_DISTANCE, BED_WIDTH, bed_width_in_rf)
    focal_bicycle= focal_length_finder(KNOWN_DISTANCE, BICYCLE_WIDTH, bicycle_width_in_rf)
    focal_bus= focal_length_finder(KNOWN_DISTANCE, BUS_WIDTH, bus_width_in_rf)
    #focal_train= focal_length_finder(KNOWN_DISTANCE, TRAIN_WIDTH, train_width_in_rf)
    focal_truck= focal_length_finder(KNOWN_DISTANCE, TRUCK_WIDTH, truck_width_in_rf)
    focal_cat= focal_length_finder(KNOWN_DISTANCE, CAT_WIDTH, cat_width_in_rf)
    focal_car= focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)
    #focal_bus= focal_length_finder(KNOWN_DISTANCE, BUS_WIDTH, bus_width_in_rf)
    #focal_bench= focal_length_finder(KNOWN_DISTANCE, BENCH_WIDTH, bench_width_in_rf)
    #focal_boat= focal_length_finder(KNOWN_DISTANCE, BOAT_WIDTH, boat_width_in_rf)
    focal_parking_meter= focal_length_finder(KNOWN_DISTANCE, PARKING_METER_WIDTH, parking_meter_width_in_rf)
    #focal_horse= focal_length_finder(KNOWN_DISTANCE, HORSE_WIDTH, horse_width_in_rf)
    focal_sheep= focal_length_finder(KNOWN_DISTANCE, SHEEP_WIDTH, sheep_width_in_rf)
    focal_cow= focal_length_finder(KNOWN_DISTANCE, COW_WIDTH, cow_width_in_rf)
    focal_elephant= focal_length_finder(KNOWN_DISTANCE, ELEPHANT_WIDTH, elephant_width_in_rf)
    focal_bear= focal_length_finder(KNOWN_DISTANCE, BEAR_WIDTH, bear_width_in_rf)
    focal_zebra= focal_length_finder(KNOWN_DISTANCE, ZEBRA_WIDTH, zebra_width_in_rf)
    focal_giraffe= focal_length_finder(KNOWN_DISTANCE, GIRAFFE_WIDTH, giraffe_width_in_rf)
    #focal_umbrella= focal_length_finder(KNOWN_DISTANCE, UMBRELLA_WIDTH, umbrella_width_in_rf)
    #focal_handbag= focal_length_finder(KNOWN_DISTANCE, HANDBAG_WIDTH, handbag_width_in_rf)
    #focal_suitcase= focal_length_finder(KNOWN_DISTANCE, SUITCASE_WIDTH, suitcase_width_in_rf)
    #focal_frisbee= focal_length_finder(KNOWN_DISTANCE, FRISBEE_WIDTH, frisbee_width_in_rf)
    #focal_snowboard= focal_length_finder(KNOWN_DISTANCE, SNOWBOARD_WIDTH, snowboard_width_in_rf)
    #focal_baseball_bat= focal_length_finder(KNOWN_DISTANCE, BASEBALL_BAT_WIDTH, baseball_bat_width_in_rf)
    #focal_baseball_glove= focal_length_finder(KNOWN_DISTANCE, BASEBALL_GLOVE_WIDTH, baseball_glove_width_in_rf)
    #focal_skateboard= focal_length_finder(KNOWN_DISTANCE, SKATEBOARD_WIDTH, skateboard_width_in_rf)
    focal_surfboard= focal_length_finder(KNOWN_DISTANCE, SURFBOARD_WIDTH, surfboard_width_in_rf)
    #focal_tennis_racket= focal_length_finder(KNOWN_DISTANCE, TENNIS_RACKET_WIDTH, tennis_racket_width_in_rf)
    focal_wineglass= focal_length_finder(KNOWN_DISTANCE, WINEGLASS_WIDTH, wineglass_width_in_rf)
    #focal_knife= focal_length_finder(KNOWN_DISTANCE, KNIFE_WIDTH, knife_width_in_rf)

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        data = object_detector(frame) 
        for d in data:
            if d[0] =='person':
                p="person"
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='cell phone':
                p="cell phone"
                distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='bottle':
                p="bottle"
                distance = distance_finder (focal_bottle, BOTTLE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='book':
                p="book"
                distance = distance_finder (focal_book, BOOK_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='chair':
                p="chair"
                distance = distance_finder (focal_chair, CHAIR_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='laptop':
                p="laptop"
                distance = distance_finder (focal_laptop, LAPTOP_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='apple':
                p="apple"
                distance = distance_finder (focal_apple, APPLE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='teddy':
                p="teddy"
                distance = distance_finder (focal_teddy, TEDDY_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='motorbike':
                p="motorbike"
                distance = distance_finder (focal_bike, MOTOR_BIKE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='cup':
                p="cup"
                distance = distance_finder (focal_cup, CUP_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='vase':
                p="vase"
                distance = distance_finder (focal_vase, VASE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='pottedplant':
                p="potted plant"
                distance = distance_finder (focal_pottedplant, POTTEDPLANT_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='car':
                p='car'
                distance = distance_finder (focal_car, CAR_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='dog':
                p="dog"
                distance = distance_finder (focal_dog, DOG_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='stopsign':
                p="stopsign"
                distance = distance_finder (focal_stopsign, STOPSIGN_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='firehydrant':
                p="firehydrant"
                distance = distance_finder (focal_firehydrant, FIREHYDRANT_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='trafficlight':
                p="trafficlight"
                distance = distance_finder (focal_trafficlight, TRAFFICLIGHT_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='keyboard':
                p="keyboard"
                distance = distance_finder (focal_keyboard, KEYBOARD_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='remote':
                p="remote"
                distance = distance_finder (focal_remote, REMOTE_WIDTH, d[1])
                x, y = d[2]
            #elif d[0] =='mouse':
            #    distance = distance_finder (focal_mouse, MOUSE_WIDTH, d[1])
            #    x, y = d[2]
            elif d[0] =='bed':
                p="bed"
                distance = distance_finder (focal_bed, BED_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='bicycle':
                p="bicycle"
                distance = distance_finder (focal_bicycle, BICYCLE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='bus':
                p="bus"
                distance = distance_finder (focal_bus, BUS_WIDTH, d[1])
                x, y = d[2]
        # elif d[0] =='train':
            #    p="train"
        #     distance = distance_finder (focal_train, TRAIN_WIDTH, d[1])
        #     x, y = d[2]
            elif d[0] =='truck':
                p="truck"
                distance = distance_finder (focal_truck, TRUCK_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='cat':
                p="cat"
                distance = distance_finder (focal_cat, CAT_WIDTH, d[1])
                x, y = d[2]
            #elif d[0] =='bus':
            #    p="bus"
            #    distance = distance_finder (focal_bus, BUS_WIDTH, d[1])
            #    x, y = d[2]
            #elif d[0] =='bench':
            #    p="bench"
            #    distance = distance_finder (focal_bench, BENCH_WIDTH, d[1])
            #    x, y = d[2]
            #elif d[0] =='boat':
            #   p="boat"
            #  distance = distance_finder (focal_boat, BOAT_WIDTH, d[1])
            # x, y = d[2]
            elif d[0] =='parking meter':
                p="parking meter"
                distance = distance_finder (focal_parking_meter, PARKING_METER_WIDTH, d[1])
                x, y = d[2]
        # elif d[0] =='horse':
            #   distance = distance_finder (focal_horse, HORSE_WIDTH, d[1])
            #   x, y = d[2]
            elif d[0] =='sheep':
                p="sheep"
                distance = distance_finder (focal_sheep, SHEEP_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='cow':
                p="cow"
                distance = distance_finder (focal_cow, COW_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='elephant':
                p="elephant"
                distance = distance_finder (focal_elephant, ELEPHANT_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='bear':
                p="bear"
                distance = distance_finder (focal_bear, BEAR_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='zebra':
                p="zebra"
                distance = distance_finder (focal_zebra, ZEBRA_WIDTH, d[1])
                x, y = d[2]
            elif d[0] =='giraffe':
                p="giraffe"
                distance = distance_finder (focal_giraffe, GIRAFFE_WIDTH, d[1])
                x, y = d[2]
            #elif d[0] =='umbrella':
            #3    p="umbrella"
            #   distance = distance_finder (focal_umbrella, UMBRELLA_WIDTH, d[1])
            #   x, y = d[2]
            #
            # elif d[0] =='handbag':
            #  p="handbag"
                #distance = distance_finder (focal_handbag, HANDBAG_WIDTH, d[1])
                #x, y = d[2]
            #elif d[0] =='suitcase':
            #   p="suitcase"
            #  distance = distance_finder (focal_suitcase, SUITCASE_WIDTH, d[1])
            # x, y = d[2]
            #elif d[0] =='frisbee':
            #    distance = distance_finder (focal_frisbee, FRISBEE_WIDTH, d[1])
            #    x, y = d[2]
        # elif d[0] =='snowboard':
            #    distance = distance_finder (focal_snowboard, SNOWBOARD_WIDTH, d[1])
            #    x, y = d[2]
            #elif d[0] =='baseball bat':
            #   distance = distance_finder (focal_baseball_bat, BASEBALL_BAT_WIDTH, d[1])
            #  x, y = d[2]
            #elif d[0] =='baseball glove':
            #    distance = distance_finder (focal_baseball_glove, BASEBALL_GLOVE_WIDTH, d[1])
            #   x, y = d[2]
            #elif d[0] =='skateboard':
            #    distance = distance_finder (focal_skateboard, SKATEBOARD_WIDTH, d[1])
            #    x, y = d[2]
            elif d[0] =='surfboard':
                p="surfboard"
                distance = distance_finder (focal_surfboard, SURFBOARD_WIDTH, d[1])
                x, y = d[2]
            #elif d[0] =='tennis racket':
            #   distance = distance_finder (focal_tennis_racket, TENNIS_RACKET_WIDTH, d[1])
            #  x, y = d[2]
            elif d[0] =='wine glass':
                p="wine glass"
                distance = distance_finder (focal_wineglass, WINEGLASS_WIDTH, d[1])
                x, y = d[2]
            #elif d[0] =='knife':
            #    distance = distance_finder (focal_knife, KNIFE_WIDTH, d[1])
            #    x, y = d[2]
            
            
            cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        cv.imshow('frame',frame)

        key = cv.waitKey(1)
        if key ==ord('q'):
            break

        output= p + "infront of you in" + str(round(distance)) +"inches"
    # import speech_recognition as sr
        import pyttsx3
        abc=pyttsx3.init() 
        speech_converting_sentence=output
        voice=abc.getProperty('voices')
        abc.setProperty('rate',120)
        abc.setProperty('volume',2.0)
        abc.setProperty('voice',voice[1].id)
        abc.say("there is a "+speech_converting_sentence)
        abc.runAndWait() 

    
    cv.destroyAllWindows()
    cap.release()

