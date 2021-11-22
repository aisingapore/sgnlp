from sgnlp.models.lif_3way_ap import Lif3WayApModel
from sgnlp.models.lif_3way_ap.modules.allennlp.model import Lif3WayApAllenNlpModel
from sgnlp.models.lif_3way_ap.modules.allennlp.predictor import Lif3WayApPredictor
from sgnlp.models.lif_3way_ap.modules.allennlp.dataset_reader import (
    Lif3WayApDatasetReader,
)

model = Lif3WayApModel.from_pretrained(
    "/Users/jonheng/aisg/sgnlp/data/lif_3way_ap/model.tar.gz",
    predictor_name="lif_3way_ap_predictor",
)

instance = {
    "context": 'Sondheim was born into a Jewish family in New York City, the son of Etta Janet ("Foxy," nee Fox; 1897-1992) and Herbert Sondheim (1895-1966). His father manufactured dresses designed by his mother. The composer grew up on the Upper West Side of Manhattan and, after his parents divorced, on a farm near Doylestown, Pennsylvania. As the only child of well-to-do parents living in the San Remo on Central Park West, he was described in Meryle Secrest\'s biography (Stephen Sondheim: A Life) as an isolated, emotionally-neglected child. When he lived in New York, Sondheim attended ECFS, the Ethical Culture Fieldston School known simply as "Fieldston." He later attended the New York Military Academy and George School, a private Quaker preparatory school in Bucks County, Pennsylvania where he wrote his first musical, By George, and from which he graduated in 1946. Sondheim spent several summers at Camp Androscoggin. He later matriculated to Williams College and graduated in 1950. He traces his interest in theatre to Very Warm for May, a Broadway musical he saw when he was nine. "The curtain went up and revealed a piano," Sondheim recalled. "A butler took a duster and brushed it up, tinkling the keys. I thought that was thrilling." When Sondheim was ten, his father (already a distant figure) left his mother for another woman (Alicia, with whom he had two sons). Herbert sought custody of Stephen but was unsuccessful. Sondheim explained to biographer Secrest that he was "what they call an institutionalized child, meaning one who has no contact with any kind of family. You\'re in, though it\'s luxurious, you\'re in an environment that supplies you with everything but human contact. No brothers and sisters, no parents, and yet plenty to eat, and friends to play with and a warm bed, you know?" Sondheim detested his mother, who was said to be psychologically abusive and projected her anger from her failed marriage on her son: "When my father left her, she substituted me for him. And she used me the way she used him, to come on to and to berate, beat up on, you see. What she did for five years was treat me like dirt, but come on to me at the same time." She once wrote him a letter saying that the "only regret [she] ever had was giving him birth." When his mother died in the spring of 1992, Sondheim did not attend her funeral and had already been estranged from her for nearly 20 years at that point. CANNOTANSWER',
    "prev_qs": ["What did he do in his early years?", "When did he graduate?"],
    "prev_ans": [
        'When he lived in New York, Sondheim attended ECFS, the Ethical Culture Fieldston School known simply as "Fieldston."',
        "He later attended the New York Military Academy and George School, a private Quaker preparatory school in Bucks County, Pennsylvania",
    ],
    "candidate": "When did he leave New York?",
}

output = model.predict_batch_json([instance])
