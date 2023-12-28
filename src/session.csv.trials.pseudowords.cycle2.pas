unit session.csv.trials.pseudowords.cycle2;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle2 }

  TCSVPseudowordsCycle2 = class(TCSVPseudowordsTrials)
    public
      constructor Create; override;
  end;

implementation

{ TCSVPseudowordsCycle2 }

constructor TCSVPseudowordsCycle2.Create;
begin
  inherited Create;
  FCycle := 2;
end;

end.
