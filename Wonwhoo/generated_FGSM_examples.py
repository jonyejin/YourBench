import os
from torchvision.utils import save_image
def generate_fgsm_example(perturbed_data, iteration, epsilon):
    # image saving process
    # if folder does not exist, make one.
    generated_image = perturbed_data[0]
    try:
        if not os.path.exists("data/epsilon " + str(epsilon)):
            os.makedirs("data/epsilon " + str(epsilon))
    except OSError:
        print ('Error: Creating directory. ' +  "data/epsilon " + str(epsilon))
    save_image(generated_image, "data/epsilon " + str(epsilon) + "/image" + str(iteration) + ".png")
    iteration = iteration + 1