unit session.csv.trials.pseudowords.cycle3;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle3 }

  TCSVPseudowordsCycle3 = class(TCSVPseudowordsTrials)
    public
      constructor Create; override;
  end;

implementation

{ TCSVPseudowordsCycle3 }

constructor TCSVPseudowordsCycle3.Create;
begin
  inherited Create;
  FCycle := 3;
end;

end.
