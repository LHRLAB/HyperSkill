"""
HyperSkill agent strategies used by the per-dataset configurations of the paper
(configs/final_configs.json). Every option is a `hypergraph.*` field of
hyperskill/config.py and is off by default.

  protocol.py      protocol_skill, protocol_answer_form   output-protocol skill learned from
                                                          the library's training failures
  prompting.py     concrete_tool_prompt, answer_suffix,   prompt, final-answer turn and
                   final_answer_turn, tool_parse_lenient,  tool-call handling
                   doc_own_only
  voting.py        vote_k, vote_temperature,              self-consistency at evaluation
                   vote_skip_refusals
  locate_pages.py  locate_pages                           doc QA: top BM25 pages after the question
  presearch.py     question_presearch                     web QA: the question pre-searched once
  video.py         time_grounding, option_check           omni multiple choice
"""
