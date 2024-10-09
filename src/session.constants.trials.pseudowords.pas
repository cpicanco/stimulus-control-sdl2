unit session.constants.trials.pseudowords;

{$mode ObjFPC}{$H+}

interface

uses session.constants.mts, session.constants.trials;

type

  TParserTrialsPseudowordsMTS = record
    ReferenceNameKey : string;
    NameKey          : string;
    SampleKey        : string;
    ComparisonsKey   : string;
    ComparisonKey    : string;
    RelationKey      : string;
    CycleKey         : string;
    ConditionKey     : string;
    CodeKey          : string;
  end;

const
  ParserTrialsPseudowordsMTS : TParserTrialsPseudowordsMTS = (
    ReferenceNameKey : HeaderReferenceName;
    NameKey          : HeaderName;
    SampleKey        : HeaderSample;
    ComparisonsKey   : HeaderComparisons;
    ComparisonKey    : HeaderComparison;
    RelationKey      : HeaderRelation;
    CycleKey         : 'Cycle';
    ConditionKey     : 'Condition';
    CodeKey          : 'Code';
  );

implementation

end.

