import openai
from openai.error import RateLimitError
import util
import train_util
from utils.config import get_dataset_config_by_name
from preamble import *
import time

openai.api_key = "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"


def get_diseases_social_dis_ner(tweet_text) -> List[str]:
    prompt = f"""
    Find all disease names in the given spanish tweets.

    Spanish Tweet:  Esta tarde mi marido y yo    viajaremos a #Bilbao a ver este interesante evento!!  Hoy nos acompañan mi artritis, mi rigidez  y mi costocondritis    gracias al deporte de esta mañana por lo menos estoy mejor! gracias por hacerlo posible @equipo_eas @Felupus #lupus https://t.co/VIZVR18thq

    Diseases(comma separated): lupus, artritis, costocondritis

    Spanish Tweet:   #Obesidad y falta de ejercicio son factores de riesgo comunes tanto para #diabetes como para #cardiopatías.
       Los países más afectados por la diabetes están también en el epicentro de la epidemia de obesidad.
       Enlace a la noticia: https://t.co/zQ9dWPi7tW
    #METAnetwork

    Diseases(comma separated): obesidad, diabetes, cardiopatías
    
    Spanish Tweet: Hoy se celebra el día mundial de la diabetes. El Servicio Canario de la Salud publica la 'Estrategia de abordaje de la diabetes mellitus en Canarias'.
#DiaMundialDeLaDiabetes #terapiaocupacionalcanarias #terapiaocupacional #coptoca #colegiate #EntreTodosPodemos https://t.co/xwhVAQSnHm

    Diseases(comma separated): diabetes, diabetes mellitus
    
    Spanish Tweet: {tweet_text}

    Diseases(comma separated):"""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
    )
    return response.choices[0].text.split(",")


def get_diseases_social_dis_ner_spanish_version(tweet_text) -> List[str]:
    prompt = f"""
    Identifique los nombres de las enfermedades en los tweets dados.

    Tweet:  Esta tarde mi marido y yo    viajaremos a #Bilbao a ver este interesante evento!!  Hoy nos acompañan mi artritis, mi rigidez  y mi costocondritis    gracias al deporte de esta mañana por lo menos estoy mejor! gracias por hacerlo posible @equipo_eas @Felupus #lupus https://t.co/VIZVR18thq

    Enfermedades: lupus, artritis, costocondritis

    Tweet:   #Obesidad y falta de ejercicio son factores de riesgo comunes tanto para #diabetes como para #cardiopatías.
       Los países más afectados por la diabetes están también en el epicentro de la epidemia de obesidad.
       Enlace a la noticia: https://t.co/zQ9dWPi7tW
    #METAnetwork

    Enfermedades: Obesidad, diabetes, cardiopatías

    Tweet: {tweet_text}

    Enfermedades:"""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
    )
    return response.choices[0].text.split(",")


def get_socialdisner_valid_predictions_from_gpt3():
    valid_samples = train_util.get_valid_samples(get_dataset_config_by_name('social_dis_ner'))
    print("num valid samples", len(valid_samples))

    gpt_completions = []
    for sample in show_progress(valid_samples):
        time.sleep(2.5)
        try:
            gpt_predictions = get_diseases_social_dis_ner(sample.text)
        except RateLimitError:
            print("rate limit error!")
            print("waiting for  secs before trying again")
            time.sleep(60)
            gpt_predictions = get_diseases_social_dis_ner(sample.text)
        gpt_completions.append((sample.id, gpt_predictions))

    util.create_json_file('./social_dis_ner_openai_output_valid.json', gpt_completions)
    # print(get_diseases_social_dis_ner(tweet))


def get_socialdisner_train_predictions_from_gpt3():
    train_samples = train_util.get_train_samples(get_dataset_config_by_name('social_dis_ner'))
    assert len(train_samples) == 5000

    gpt_completions = []
    for i, sample in show_progress(enumerate(train_samples), total=len(train_samples)):
        time.sleep(2.5)
        try:
            gpt_predictions = get_diseases_social_dis_ner(sample.text)
        except RateLimitError:
            print("rate limit error!")
            print("waiting for 60 secs before trying again")
            time.sleep(60)
            gpt_predictions = get_diseases_social_dis_ner(sample.text)
        gpt_completions.append((sample.id, gpt_predictions))
        if (i % 100) == 0:
            print("Storing intermediate results")
            util.create_json_file('./social_dis_ner_openai_output_train.json', gpt_completions)

    util.create_json_file('./social_dis_ner_openai_output_train.json', gpt_completions)
