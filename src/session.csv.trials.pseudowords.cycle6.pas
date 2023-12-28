unit session.csv.trials.pseudowords.cycle6;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle6 }

  TCSVPseudowordsCycle6 = class(TCSVPseudowordsTrials)
    public
      constructor Create; override;
  end;

implementation

{ TCSVPseudowordsCycle6 }

constructor TCSVPseudowordsCycle6.Create;
begin
  inherited Create;
  FCycle := 6;
end;

end.
