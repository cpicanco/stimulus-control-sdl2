{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials.pseudowords.cycle5;

{$mode ObjFPC}{$H+}

interface

uses SysUtils, session.csv.trials.pseudowords;

type

  { TCSVPseudowordsCycle5 }

  TCSVPseudowordsCycle5 = class(TCSVPseudowordsTrials)
    public
      constructor Create(ASource: string); override;
  end;

implementation

{ TCSVPseudowordsCycle5 }

constructor TCSVPseudowordsCycle5.Create(ASource: string);
begin
  inherited Create(ASource);
  FCycle := 5;
end;

end.
