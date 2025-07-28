# https://huggingface.co/EmergentMethods/Qwen3-4B-BiasExpert

# Example usage of the Qwen3-4B-BiasExpert model to analyze media bias in a news article.

# Import dependencies
from vllm import LLM

# prepare model
llm = LLM(
    model="EmergentMethods/Qwen3-4B-BiasExpert",
    trust_remote_code=True,
    max_model_len=24000,
    max_num_seqs=2,
)

# Create sampling params object
sampling_params = llm.get_default_sampling_params()
sampling_params.max_tokens = 20000

# https://huggingface.co/Qwen/Qwen3-4B#best-practices
sampling_params.temperature = 0.6
sampling_params.top_p = 0.95
sampling_params.top_k = 20
sampling_params.min_p = 0

# Put the content of your article in the variable below.
article_text = """
Title\n\nContent
"""

messages = [
    {"role": "user", "content": """You are an ethical, expert journalist whose sole source of information is the news article provided to you. Your task is to analyze the article for specific types of media bias.

## Task
Analyze the provided news article to identify and evaluate 18 specific types of media bias. Your analysis should be thorough, evidence-based, and rely solely on the content of the article provided.

## Analysis Process

1. **Initial Review**: Read the entire article carefully to identify main entities (people, organizations, groups, concepts) and note how each is characterized or framed.

2. **Headline Analysis**: 
   - Identify the headline (use the first phrase/sentence of the article if not clearly marked as a headline)
   - Compare headline to actual content
   - Look for emotional language or exaggeration
   - Identify if headline accurately represents the article

3. **Language Assessment**:
   - Check for subjective adjectives and loaded terms
   - Identify emotional language and tone
   - Note labels applied to individuals or groups

4. **Source and Attribution Review**:
   - Identify sources and their diversity
   - Check for missing attributions or vague references
   - Evaluate if opposing viewpoints are represented fairly

5. **Fact vs. Opinion Separation**:
   - Distinguish between factual statements and opinions
   - Identify opinion statements presented as facts
   - Note unsupported claims or logical fallacies

6. **Contextual Analysis**:
   - Identify missing context or background information
   - Check for cherry-picked data or statistics
   - Note historical or social context omissions

7. **Bias Evaluation**: Evaluate the text against each of the 18 types of bias using their definitions and examples. A text may contain several different types of bias, with some types being more general and others more detailed, which may result in overlap. Always identify all types of bias present according to the provided definitions.

8. **Bias Level Assignment**: For each bias type, assign a level:
   - **None**: No detectable bias of this type is present
   - **Low**: Minor signs of bias that don't significantly affect overall neutrality
   - **Moderate**: Noticeable bias that somewhat influences framing or perception
   - **High**: Dominant bias that strongly shapes the narrative or portrayal

9. **Evidence-Based Reasoning**: If bias is detected (Low, Moderate, or High), *you must cite specific words, phrases, sentences, or omissions from the article text* as evidence. Explain *how* this evidence demonstrates the specific bias type.

10. **Improvement Suggestions**: For each bias, optionally provide one or more suggestions with:
    - Description: How to fix or balance this bias
    - Reasoning: Why that would help

11. **Summary Generation**: Create a comprehensive summary ("bias_summary" JSON entry) of overall bias patterns, synthesizing the most prominent biases found or stating if the article appears largely unbiased.

12. **Formatting**: Format your analysis according to the provided JSON schema (title of the schema: Bias Analysis Schema).


## 18 Types of Bias (Definitions and examples)

### 1. Political Bias
**Definition:** Content that explicitly or implicitly favors or criticizes a specific political viewpoint, party, or ideology.
**Example:** "The radical left continues to sabotage the economy."
**Analysis guidance:** Look for partisan language, uneven treatment of political figures/parties, or ideological framing that presents one political perspective as superior.

### 2. Gender Bias
**Definition:** Content that reinforces stereotypes, shows prejudice, or makes generalizations based on gender.
**Example:** "The female engineer surprisingly solved the problem."
**Analysis guidance:** Identify instances where gender is unnecessarily mentioned, where stereotypes are reinforced, or where different standards are applied based on gender.

### 3. Cultural/Ethnicity Bias
**Definition:** Content that unfairly portrays, generalizes, or stereotypes ethnic or cultural groups.
**Example:** "Immigrants are taking away local jobs."
**Analysis guidance:** Look for generalizations about ethnic groups, uneven portrayal of cultures, or language that "others" certain groups.

### 4. Age Bias
**Definition:** Content that unfairly stereotypes or discriminates based on age.
**Example:** "Older employees rarely adapt to new technology."
**Analysis guidance:** Identify age-based generalizations, stereotypes about generation groups, or dismissive attitudes toward certain age groups.

### 5. Religion Bias
**Definition:** Content that unfairly stereotypes or discriminates based on religious beliefs.
**Example:** "Muslim neighborhoods are often hotspots of radicalism."
**Analysis guidance:** Look for generalizations about religious groups, uneven treatment of different faiths, or language that portrays certain religions in a consistently negative light.

### 6. Disability Bias
**Definition:** Content that portrays individuals with disabilities or mental health conditions in a negative, stereotypical, or dehumanizing way, often using outdated or offensive language.
**Example:** "This facility is for retarded individuals."
**Analysis guidance:** Identify outdated terminology, narratives that present disability as shameful or abnormal, or representations that define people primarily by their disabilities.

### 7. Statement Bias (Labelling and Word Choice)
**Definition:** The use of loaded language or partisan labeling that reveals the author's perspective on the topic, often presenting one side of an issue as the only legitimate view.
**Example:** Using "pro-life" vs. "anti-abortion" or "gender-affirming care" vs. "sex reassignment procedure."
**Analysis guidance:** Identify loaded terms, politically charged labeling, and word choices that reveal underlying assumptions or ideological perspectives.

### 8. Unsubstantiated or Illogical Claims
**Definition:** Making assertions without supporting evidence or using flawed reasoning to lead readers to misleading conclusions.
**Example:** "The senator's absence clearly shows he doesn't care about the crisis."
**Analysis guidance:** Look for claims without citations or evidence, logical fallacies, or conclusions that do not logically follow from presented evidence.

### 9. Slant (Bias by Omission)
**Definition:** Highlighting certain angles or information while downplaying or omitting other relevant perspectives, preventing readers from seeing the full picture.
**Example:** Reporting only positive outcomes of a policy while ignoring documented drawbacks.
**Analysis guidance:** Identify what perspectives or facts are missing that would provide a more complete understanding of the issue.

### 10. Source Selection Bias
**Definition:** Choosing sources that support a predetermined narrative rather than seeking diverse viewpoints for balance.
**Example:** Interviewing only company representatives about an environmental disaster without including affected residents or independent experts.
**Analysis guidance:** Examine the range of perspectives represented through quoted sources and whether key stakeholders are missing.

### 11. Omission of Source Attribution
**Definition:** Making claims without proper attribution or using vague, unspecified sources.
**Example:** Using phrases like "according to sources," "critics say," or "experts believe" without specificity.
**Analysis guidance:** Look for claims that lack clear attribution or rely on anonymous or generalized sources without justification.

### 12. Spin
**Definition:** Using rhetoric techniques to create a more memorable or emotionally resonant story, often at the expense of objectivity.
**Example:** Using dramatic framing or emotional language to create a narrative beyond the basic facts.
**Analysis guidance:** Identify language choices that go beyond factual reporting to create a particular impression or emotion.

### 13. Sensationalism
**Definition:** Exaggerating information to provoke an emotional reaction, often to increase engagement.
**Example:** "Bloodbath at the debate stage last night!"
**Analysis guidance:** Look for hyperbolic language, emotional framing, or exaggeration of events beyond their actual significance.

### 14. Negativity Bias
**Definition:** Emphasizing negative aspects of events or framing stories in a consistently negative light.
**Example:** "The country is collapsing under the weight of failed leadership."
**Analysis guidance:** Identify whether negative aspects are disproportionately emphasized compared to positive or neutral information.

### 15. Subjective Adjectives
**Definition:** Using qualifying adjectives that characterize or attribute specific properties to subjects, inserting the writer's judgment rather than letting readers form their own.
**Example:** "The disturbing trend in education continues."
**Analysis guidance:** Look for adjectives that reveal the writer's perspective or attempt to frame how readers should interpret information.

### 16. Ad Hominem/Mudslinging
**Definition:** Making unfair or insulting accusations about a person's character rather than addressing their ideas or arguments.
**Example:** "He's a clown with no experience or credibility."
**Analysis guidance:** Identify attacks on personal characteristics, motives, or backgrounds that distract from substantive discussion.

### 17. Mind Reading
**Definition:** Asserting knowledge about a person's thoughts, intentions, or motives without evidence.
**Example:** "She clearly intended to undermine the election."
**Analysis guidance:** Look for claims about what someone thought, felt, or intended without direct evidence or quotes.

### 18. Opinion-as-Fact
**Definition:** Presenting subjective judgments or interpretations as if they were objective facts.
**Example:** "This policy is proof that the government doesn't care about citizens."
**Analysis guidance:** Identify opinions or interpretations that are presented without qualifying language that would mark them as subjective.


## Schema / Response Format

The response must be JSON following this schema.


Schema: 
{
  "title": "Bias Analysis Schema",
  "type": "object",
  "properties": {
    "bias_summary": {
      "type": "string"
    },
    "bias_analysis": {
      "type": "object",
      "properties": {
        "political": {
          "$ref": "#/definitions/bias_entry"
        },
        "gender": {
          "$ref": "#/definitions/bias_entry"
        },
        "ethnic_cultural": {
          "$ref": "#/definitions/bias_entry"
        },
        "age": {
          "$ref": "#/definitions/bias_entry"
        },
        "religion": {
          "$ref": "#/definitions/bias_entry"
        },
        "disability": {
          "$ref": "#/definitions/bias_entry"
        },
        "statement": {
          "$ref": "#/definitions/bias_entry"
        },
        "unsubstantiated_illogical_claims": {
          "$ref": "#/definitions/bias_entry"
        },
        "slant": {
          "$ref": "#/definitions/bias_entry"
        },
        "source_selection": {
          "$ref": "#/definitions/bias_entry"
        },
        "omission_of_source_attribution": {
          "$ref": "#/definitions/bias_entry"
        },
        "spin": {
          "$ref": "#/definitions/bias_entry"
        },
        "sensationalism": {
          "$ref": "#/definitions/bias_entry"
        },
        "negativity": {
          "$ref": "#/definitions/bias_entry"
        },
        "subjective_adjectives": {
          "$ref": "#/definitions/bias_entry"
        },
        "mudslinging": {
          "$ref": "#/definitions/bias_entry"
        },
        "mind_reading": {
          "$ref": "#/definitions/bias_entry"
        },
        "opinion_as_fact": {
          "$ref": "#/definitions/bias_entry"
        }
      },
      "additionalProperties": false
    }
  },
  "definitions": {
    "bias_entry": {
      "type": "object",
      "properties": {
        "level": {
          "type": "string",
          "enum": [
            "High",
            "Moderate",
            "Low",
            "None"
          ]
        },
        "reasoning": {
          "type": "string"
        },
        "suggestions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "description": {
                "type": "string"
              },
              "reasoning": {
                "type": "string"
              }
            },
            "required": [
              "description",
              "reasoning"
            ],
            "additionalProperties": false
          }
        }
      },
      "required": [
        "level",
        "reasoning"
      ],
      "additionalProperties": false
    }
  },
  "required": [
    "bias_summary",
    "bias_analysis"
  ],
  "additionalProperties": false
}



Example:

{
  "bias_summary": "...",
  "bias_analysis": {
    "political": { "level": "Moderate", "reasoning": "...", "suggestions": [] },
    "gender": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "ethnic_cultural": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "age": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "religion": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "disability": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "statement": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "unsubstantiated_illogical_claims": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "slant": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "source_selection": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "omission_of_source_attribution": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "spin": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "sensationalism": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "negativity": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "subjective_adjectives": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "mudslinging": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "mind_reading": { "level": "Low", "reasoning": "...", "suggestions": [] },
    "opinion_as_fact": { "level": "Low", "reasoning": "...", "suggestions": [] }
  }
}




## Article
\n\n
""" + article_text}
]


def print_outputs(outputs):
    print("\nGenerated Outputs:\n" + "-" * 80)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)

outputs = llm.chat(messages, sampling_params, use_tqdm=False)
print_outputs(outputs)