import xml.etree.ElementTree as ET

text = "Previous chemotherapy and exposure to radiation may increase the risk of developing ALL. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk factors doesn’t mean that you will not get cancer. Talk with your doctor if you think you may be at risk. Possible risk factors for ALL include the following: Being male. Being White. Being older than 70. Past treatment with chemotherapy or radiation therapy. Being exposed to high levels of radiation in the environment (such as nuclear radiation). Having certain genetic disorders, such as Down syndrome. Signs and symptoms of adult ALL include fever, feeling tired, and easy bruising or bleeding. The early signs and symptoms of ALL may be like the flu or other common diseases. Check with your doctor if you have any of the following: Weakness or feeling tired. Fever or night sweats. Easy bruising or bleeding. Petechiae (flat, pinpoint spots under the skin, caused by bleeding). Shortness of breath. Weight loss or loss of appetite. Pain in the bones or stomach. Pain or feeling of fullness below the ribs. Painless lumps in the neck, underarm, stomach, or groin. Having many infections. These and other signs and symptoms may be caused by adult acute lymphoblastic leukemia or by other conditions. Tests that examine the blood and bone marrow are used to diagnose adult ALL."

def extract_answers(xml_data):
    answers = []
    root = ET.fromstring(xml_data)
    answer_elements = root.findall(".//Answer")

    for answer_element in answer_elements:
    	answer_text = answer_element.text.strip()
    	answers.append(answer_text)

    return answers


def get_span(xml_file):

    # get the text from the parser here
    # text = parse()

    with open(xml_file, 'r') as f:
        xml_data = f.read()
    
    answers = extract_answers(xml_data)
    refined_answers = []

    spans = []
    for answer in answers:
        answer = answer.replace('-',' ')
        refined_answer = " ".join(answer.split())
        result = text.find(refined_answer)
        if(result!=-1):
            i, j = result, result + len(refined_answer)
            spans.append((i,j))
        else:
            spans.append((-1,-1))
    return spans

if __name__ == "__main__":
    xml_file = 'test_data.xml'
    spans = get_span(xml_file)
    for span in spans:
        if(span[0]!=-1):
            print("Span is:", span[0], span[1])
            print(text[span[0]:span[1]])

