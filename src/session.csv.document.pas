{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.document;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, csvdocument;

type

  { TCustomCSVDocument }

  TCustomCSVDocument = class(TCSVDocument)
  private
    FSkipHeader : boolean;
    FCurrentIndex: integer;
  published
    constructor Create; override;
    property CurrentIndex : integer read FCurrentIndex write FCurrentIndex;
    property SkipHeader : boolean read FSkipHeader write FSkipHeader;
  end;

implementation

{ TCustomCSVDocument }

constructor TCustomCSVDocument.Create;
begin
  inherited Create;
  FSkipHeader := True;
  FCurrentIndex := 0;
end;

end.

