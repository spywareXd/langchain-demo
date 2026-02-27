import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def main():
    print("Hello from langchain-course!")

    text="""   EsDeeKid is a British rapper from Liverpool, England.[1] He released his debut album, Rebel, in June 2025.[2][3]

His music is characterised by his thick Scouse accent[4] and draws inspiration from cloud rap.[5] EsDeeKid's identity is unknown; he covers his face with a balaclava and keeps personal information, such as his age, private. He has mentioned on his Twitter that he still lives in his council house.[1][5]

As of January 2026, EsDeeKid's single "Phantom" (with Rico Ace) has accumulated more than 170 million streams. His album Rebel has charted in multiple European countries,[6][7][8] as well as peaked at number 23 on the US Billboard 200,[9] despite being released nearly five months prior to its debut on the chart.[10] EsDeeKid has been described by Dazed as "the fastest-growing artist in the world right now".[11]

Background and persona
EsDeeKid originates from Liverpool and his regional identity is a key part of his appeal.[1] He performs with his face concealed by a balaclava and has not publicly disclosed his real name, age or detailed biographical information.[1][5] Both interviews and reviews note that he rarely breaks character in public, maintaining a mysterious, anonymous persona centred on dark visuals and minimal personal exposure.[11][12]

His accent has been singled out by critics as one of his defining characteristics, helping to distinguish him within the wider field of UK rap.[1][4] The Guardian's "Add to playlist" column described him as having a "magnificent Scouse accent" and "entertainingly debauched lyrics", noting his rapid rise from Liverpool's underground scene.[1] Dazed interviewed fans who have highlighted his menacing stage presence, all-black clothing and masked image as central to his appeal.[12]

Career
2023–2024: Early releases and recognition
EsDeeKid's first known song was a feature on "Altered" by DualSpines, released in 2023.[11] He began releasing standalone music in May 2024, releasing a remix of "Black Beatles" by Rae Sremmurd on SoundCloud.[13] Other non-album singles he released include "Apathy", "Slurricane" (with Fakemink), "Number (N)ine Freestyle", "Gracias", "Palaces" (with Rico Ace), "Tapped In" (with SINN6R), "Ferragamo", "Bally" (with Rico Ace), "Little Kidda" and "Warmin' Up".[7]

His 2024 single "Bally" featuring Rico Ace was highlighted in Pitchfork's "The Ones" column, where Alphonse Pierre described it as one of the year's standout tracks and named EsDeeKid one of the year's breakout stars.[14] The song received international success and established his partnership with Rico Ace, who would later appear on some of his best-known tracks.

2025: Rebel and commercial breakthrough
Main article: Rebel (EsDeeKid album)
His debut album, Rebel, was released on 20 June 2025 through Lizzy Records.[15] The project included the singles "LV Sandals", "5am", and "Phantom" alongside 8 other original tracks.[7] Rebel charted in the United Kingdom, Ireland and several other European countries, reaching number 16 on the UK Albums Chart and appearing on national album charts in Austria, Belgium, Finland, Lithuania, the Netherlands, Norway, Sweden and Switzerland.[16][8][17][18][19] His monthly listeners on streaming platforms such as Spotify and Youtube, also increased significantly.[7][20]

As of December 2025, EsDeekid's most listened track on Spotify is "Phantom" (featuring Rico Ace). Since its release in March 2025,[6] it has amassed over 240 million streams.[7]

Live performances
Following the release of Rebel, EsDeeKid launched his European tour in 2025, performing in the United Kingdom and continental Europe. Dazed described the run of dates as "career-defining", reporting that he rounded off the tour with a show at the Electric Ballroom in Camden Town, London.[12] The Face described his London date on the Rebel tour as "legendary", highlighting intense mosh pits, a riotous atmosphere and fans showing up early to see his first major headline show in the city.[21]

According to various sources, his shows prominently feature a heavy, almost rock-influenced live energy, with dark lighting, frequent moshing and EsDeeKid often appearing as a shadowy figure at the centre of the stage, underscoring his mysterious persona.[21][12]
    """

    summary_template= """ This is the information provided of a person: {info}. I want you to do 2 things :
                            1. Summarize the information under 50 words
                            2, Display any 3 interesting facts of this person"""

    summary_prompt=PromptTemplate( input_variables=["info"]   , template=summary_template)

    llm=ChatOllama(model="llama3")

    chain = summary_prompt | llm

    response=chain.invoke({"info": text})

    print(response.content)









if __name__ == "__main__":
    main()
