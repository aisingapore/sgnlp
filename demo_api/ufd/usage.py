import torch
import torch.nn.functional as F

from sgnlp.models.ufd import (
    UFDModelBuilder,
    UFDPreprocessor,
)

# Constants
DEVICE = torch.device("cpu")

# Load model and preprocessor
preprocessor = UFDPreprocessor()
model_builder = UFDModelBuilder(
    source_domains=["music"], target_languages=["de"], target_domains=["books"]
)
models = model_builder.build_model_group()
model = models["music_de_books"]


text = (
    "Deutschland, ein kurzes, langes Glück Zuallererst: Die Geschichte ist vordergründig schnell erzählt. "
    "Eine Frau, ein Mann im Zentrum des Geschehens. Beide nicht gescheitert. Beide aber auch nicht auf der "
    "Siegerseite. Beide mittendrin im Leben. Inmitten Deutschlands. In der Mittelschicht und mittelalt. Die "
    "Beiden entdecken sich, sie lieben sich, sie hören nicht auf, ihr Leben zu leben. Sie resignieren nicht. "
    "Klingt doch langweilig, oder? Klingt nach Abziehbild der Realität. Klingt keineswegs nach dem Stoff, aus "
    "dem die (Roman-)träume sind. Doch halt: Das ist ein leises Buch, ein genaues Buch. Das Außergewöhnliche "
    "an diesem 540-Seiten-Roman ist, dass der Autor es schafft, nicht nur das Leben der Beiden, sondern damit "
    "auch unser aller Leben im Deutschland des ausgehenden 20. Jahrhundertes und des beginnenden 21. "
    "Jahrhunderts mit einer Präzision sondersgleichen zu zeichnen. Der junge Autor Stephan Thome hat sich aus "
    "der Ferne - er lebt in Taiwan - ganz nahe herangezoomt, einen winzigen Ausschnitt angesehen. Oder "
    "genauer: Mehrere winzige Ausschnitte, denn er fokussiert ein alle sieben Jahre stattfindendes "
    "kleinstädtisches Fest namens Grenzgang und blendet die Zeiten dazwischen aus diese erstehen im Kopf des "
    "Lesers. Bis in die feinsten Verästelungen hinein spürt er den Dingen, den Menschen nach, versetzt sich "
    "in sie hinein. Sieht genau zu, sieht genau hin, bleibt in jeder Sekunde kompromisslos, ergreift keine "
    "Partei für niemanden, bleibt stehen, erzählt mit einer Unerbittlichkeit, die einzigartig ist, erfasst "
    "kleinste Erschütterungen, wie ein Seismograph. Der erste Kuss, der erste Sex, die Annäherungen, die "
    "Trennungen, die kleineren Siege inmitten der größeren Niederlagen. Diesen Roman in wenigen Zeilen zu "
    "würdigen, das will schwer gelingen. Meisterlich die Dialoge, meisterlich die Figurenzeichnungen aller "
    "Protagonisten, auch jener, die am Rande des Romanes stehen, meisterlich die Idee, den Roman in "
    "Siebenjahresschritten zu erzählen: Mit welcher Leichtigkeit der Autor es dadurch erreicht, dass sich im "
    "Kopf des Lesers die scheinbar fehlenden Teile des Puzzles zum Ganzen fügen: Das ist große Kunst. Stephan "
    "Thome hat Deutschland gezeichnet. Ohne jemals beleidigend zu werden, ohne anzuklagen, ohne jedweden "
    "Zynismus. Genauer kann man dieses Land nicht erfassen. Dass vieles weh tut, wenn man genau hinsieht und "
    "hineinsieht in die Vorgärten dieses Landes, das versteht sich von selbst. Am Schluss aber bleibt eines, "
    "im Leben wie in diesem Werk: Die Hoffnung, die Liebe. Und es bleibt ein großer Roman, der noch lange "
    "Bestand haben wird."
)

text_features = preprocessor([text])
output = model(**text_features)
logits_probabilities = F.softmax(output.logits, dim=1)
