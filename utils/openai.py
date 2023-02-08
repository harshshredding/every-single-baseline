import openai
from preamble import *

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
