unit session.csv.trials.pseudowords.cycle1;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle1 }

  TCSVPseudowordsCycle1 = class(TCSVPseudowordsTrials)
    public
      constructor Create; override;
  end;

implementation

{ TCSVPseudowordsCycle1 }

constructor TCSVPseudowordsCycle1.Create;
begin
  inherited Create;
  FCycle := 1;
end;

end.


