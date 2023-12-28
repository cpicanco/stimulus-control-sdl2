unit session.csv.trials.pseudowords.cycle4;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle4 }

  TCSVPseudowordsCycle4 = class(TCSVPseudowordsTrials)
    public
      constructor Create; override;
  end;

implementation

{ TCSVPseudowordsCycle4 }

constructor TCSVPseudowordsCycle4.Create;
begin
  inherited Create;
  FCycle := 4;
end;

end.
