unit session.constants.trials.dapaap;

{$mode ObjFPC}{$H+}

interface

uses session.constants.mts, session.constants.trials;

type
  TParserTrialsDAPAAP = record
    ReferenceNameKey : string;
    NameKey          : string;
    SampleKey        : string;
    ComparisonKey    : string;
    RelationKey      : string;
    SubsetKey        : string;
    HasPromptKey     : string;
    TotalLoopsKey    : string;
  end;

const
  ParserTrialsDAPAAP : TParserTrialsDAPAAP = (
    ReferenceNameKey : HeaderReferenceName;
    NameKey          : HeaderName;
    SampleKey        : HeaderSample;
    ComparisonKey    : HeaderComparison;
    RelationKey      : HeaderRelation;
    SubsetKey        : 'Subset';
    HasPromptKey     : HeaderHasPrompt;
    TotalLoopsKey    : 'TotalLoops';
  );

implementation

end.

