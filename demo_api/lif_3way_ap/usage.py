from sgnlp.models.lif_3way_ap import Lif3WayApModel
from sgnlp.models.lif_3way_ap.modules.allennlp.model import Lif3WayApAllenNlpModel
from sgnlp.models.lif_3way_ap.modules.allennlp.predictor import Lif3WayApPredictor
from sgnlp.models.lif_3way_ap.modules.allennlp.dataset_reader import (
    Lif3WayApDatasetReader,
)

model = Lif3WayApModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/lif_3way_ap/model.tar.gz",
    predictor_name="lif_3way_ap_predictor",
)

inputs = [
    {
        "context": "Acuff was born on September 15, 1903 in Maynardville, Tennessee, to Ida (nee Carr) and Simon E. Neill Acuff, the third of their five children. The Acuffs were a fairly prominent family in Union County. Roy's paternal grandfather, Coram Acuff, had been a Tennessee state senator, and his maternal grandfather was a local physician. Roy's father was an accomplished fiddler and a Baptist preacher, his mother was proficient on the piano, and during Roy's early years the Acuff house was a popular place for local gatherings. At such gatherings, Roy would often amuse people by balancing farm tools on his chin. He also learned to play the harmonica and jaw harp at an early age. In 1919, the Acuff family relocated to Fountain City (now a suburb of Knoxville), a few miles south of Maynardville. Roy attended Central High School, where he sang in the school chapel's choir and performed in \"every play they had.\" His primary passion, however, was athletics. He was a three-sport standout at Central and, after graduating in 1925, was offered a scholarship to Carson-Newman University but turned it down. He played with several small baseball clubs around Knoxville, worked at odd jobs, and occasionally boxed. In 1929, Acuff tried out for the Knoxville Smokies, a minor-league baseball team then affiliated with the New York Giants (now the San Francisco Giants). A series of collapses in spring training following a sunstroke, however, ended his baseball career. The effects left him ill for several years, and he suffered a nervous breakdown in 1930. \"I couldn't stand any sunshine at all,\" he later recalled. While recovering, Acuff began to hone his fiddle skills, often playing on the family's front porch after the sun went down. His father gave him several records of regionally renowned fiddlers, such as Fiddlin' John Carson and Gid Tanner, which were important influences on his early style.",
        "prev_qs": ["where was he born?"],
        "prev_ans": ["Maynardville, Tennessee,"],
        "candidate": "who were his parents",
    },
    {
        "context": 'Sondheim was born into a Jewish family in New York City, the son of Etta Janet ("Foxy," nee Fox; 1897-1992) and Herbert Sondheim (1895-1966). His father manufactured dresses designed by his mother. The composer grew up on the Upper West Side of Manhattan and, after his parents divorced, on a farm near Doylestown, Pennsylvania. As the only child of well-to-do parents living in the San Remo on Central Park West, he was described in Meryle Secrest\'s biography (Stephen Sondheim: A Life) as an isolated, emotionally-neglected child. When he lived in New York, Sondheim attended ECFS, the Ethical Culture Fieldston School known simply as "Fieldston." He later attended the New York Military Academy and George School, a private Quaker preparatory school in Bucks County, Pennsylvania where he wrote his first musical, By George, and from which he graduated in 1946. Sondheim spent several summers at Camp Androscoggin. He later matriculated to Williams College and graduated in 1950. He traces his interest in theatre to Very Warm for May, a Broadway musical he saw when he was nine. "The curtain went up and revealed a piano," Sondheim recalled. "A butler took a duster and brushed it up, tinkling the keys. I thought that was thrilling." When Sondheim was ten, his father (already a distant figure) left his mother for another woman (Alicia, with whom he had two sons). Herbert sought custody of Stephen but was unsuccessful. Sondheim explained to biographer Secrest that he was "what they call an institutionalized child, meaning one who has no contact with any kind of family. You\'re in, though it\'s luxurious, you\'re in an environment that supplies you with everything but human contact. No brothers and sisters, no parents, and yet plenty to eat, and friends to play with and a warm bed, you know?" Sondheim detested his mother, who was said to be psychologically abusive and projected her anger from her failed marriage on her son: "When my father left her, she substituted me for him. And she used me the way she used him, to come on to and to berate, beat up on, you see. What she did for five years was treat me like dirt, but come on to me at the same time." She once wrote him a letter saying that the "only regret [she] ever had was giving him birth." When his mother died in the spring of 1992, Sondheim did not attend her funeral and had already been estranged from her for nearly 20 years at that point.',
        "prev_qs": ["What did he do in his early years?", "When did he graduate?"],
        "prev_ans": [
            'When he lived in New York, Sondheim attended ECFS, the Ethical Culture Fieldston School known simply as "Fieldston."',
            "He later attended the New York Military Academy and George School, a private Quaker preparatory school in Bucks County, Pennsylvania",
        ],
        "candidate": "When did he leave New York?",
    },
]


# batch predict example
output = model.predict_batch_json(inputs)

# single instance predict example
output = model.predict_json(inputs[0])
