unit session.csv.trials.pseudowords.cycle5;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle5 }

  TCSVPseudowordsCycle5 = class(TCSVPseudowordsTrials)
    public
      constructor Create; override;
  end;

implementation

{ TCSVPseudowordsCycle5 }

constructor TCSVPseudowordsCycle5.Create;
begin
  inherited Create;
  FCycle := 5;
end;

end.
