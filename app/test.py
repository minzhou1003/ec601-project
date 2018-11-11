import os
from subprocess import call, Popen, PIPE

parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
darknet_path = os.path.join(parent_path, 'yolo_model', 'darknet')
uploads_path = os.path.join(os.getcwd(), 'uploads')

def predict(input_image_path, threh=0.01):
    current_path = os.getcwd()
    os.chdir(darknet_path)
    p = Popen(['./darknet', 'detector', 'test', 
        '../cfg/rsna.data', '../cfg/rsna_yolov3.cfg_test', 
        '../backup/rsna_yolov3_900.weights', input_image_path, 
        '-thresh', f'{threh}'], stdout=PIPE)
    output = p.communicate()[0]
    os.chdir(current_path)
    return output

# predict(f'{uploads_path}/pos_test339.jpg', 0.01)



        



output = "b'/Users/minzhou/Desktop/EC601/ec601-project/app/uploads/pos_test339.jpg: Predicted in 27.170969 seconds.\npneumonia: 33%\npneumonia: 9%\npneumonia: 6%\npneumonia: 5%\npneumonia: 2%\npneumonia: 1%\npneumonia: 1%\npneumonia: 1%\npneumonia: 1%\npneumonia: 1%\npneumonia: 1%\npneumonia: 1%\npneumonia: 1%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\npneumonia: 0%\n'"


output = parse_prediction_result(output)
